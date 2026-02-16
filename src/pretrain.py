import torch
import torch.distributed as dist
import math
import time
import gc
from torch.nn.parallel import DistributedDataParallel
from model import FiberGPT, FiberGPTConfig
from utils.common import compute_init, compute_cleanup, print0, save_model
from utils.attention import fa3_available
from utils.dataloader import DDGPretrain
from utils.muon import Muon, DistributedMuon
from utils.tokenizer import tokenizer


device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
# Are we going distribtued?
distributed, rank, local_rank, world_size, device = compute_init(device_type)
master_process = rank == 0


# HYPERPARAMETERS
CONTEXT_LENGTH = FiberGPTConfig.context_length
BATCH_SIZE = 1_048_576  # ~1M in tokens
NO_OF_BATCH = BATCH_SIZE // (world_size * CONTEXT_LENGTH)
NO_OF_BATCH_PER_DEVICE = 64  # Change this based on GPU VRAM (enough for H100)
GRAD_ACCUM_STEPS = NO_OF_BATCH // NO_OF_BATCH_PER_DEVICE

# TRAINING HYPERPARAMS
MAX_ITER = 10_000  # Roughly enough to go through the entire dataset

# Warmup on 500M tokens -> Stabilize
# Constnat on 4B tokens -> Learn main patterns
# Warmdown on 5.5B tokens -> Fine-tune and converge
WARMUP_ITER_RATIO = 0.05
COOLDOWN_ITER_RATIO = 0.55
FINAL_LR_RATIO = 0.1
MUON_MOMENTUM_MAX = 0.95
MUON_MOMENTUM_MIN = 0.85
MUON_MOMENTUM_WARMUP_ITER_RATIO = 0.05
MUON_MOMENTUM_COOLDOWN_ITER_RATIO = 0.01

# Sample, eval and checkpoint
SAMPLE_EVERY = 500
EVAL_EVERY = 500
EVAL_STEPS = 5
CHECKPOINT_EVERY = 500


# Warn if FA3 is not available.
if not fa3_available:
    print0(
        'WARNING: Flash Attention 3 is not available, using SDPA fallback.'
        ' Training will be inefficient!'
    )


# Training states
autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)


# Compile the model
orig_model = FiberGPT(FiberGPTConfig()).to(device)
compiled_model = torch.compile(orig_model, dynamic=False, fullgraph=True)
model = compiled_model

# Wrap model in DDP
if distributed:
    model = DistributedDataParallel(
        model,
        device_ids=[ local_rank ],
        broadcast_buffers=False,
        gradient_as_bucket_view=True
    )


# Optimizers
optims = orig_model.setup_optimizers(
    muon_momentum=MUON_MOMENTUM_MIN
)  # Mostly use the default optimizer parameters

def optim_step():
    for optim in optims:
        optim.step()

def optim_zero_grad():
    for optim in optims:
        optim.zero_grad(set_to_none=True)

def optim_update_params(lrm: float, momentum: float):
    for optim in optims:
        for group in optim.param_groups:
            group['lr'] = group['initial_lr'] * lrm

            if isinstance(optim, (Muon, DistributedMuon)):
                group['momentum'] = momentum


# Initialize dataloaders for train and val
train_loader = DDGPretrain(
    'data/fineweb10b/fineweb_train_*.bin',
    NO_OF_BATCH_PER_DEVICE, CONTEXT_LENGTH, rank, world_size
)
eval_loader = DDGPretrain(
    'data/fineweb10b/fineweb_val_*.bin',
    NO_OF_BATCH_PER_DEVICE, CONTEXT_LENGTH, rank, world_size
)


# Learning rate scheduling (linear warmup, constant, cosine decay)
def get_lr_multiplier(step: int):
    warmup_iter = round(WARMUP_ITER_RATIO * MAX_ITER)
    cooldown_iter = round(COOLDOWN_ITER_RATIO * MAX_ITER)

    # Linear warmup
    if step < warmup_iter:
        return (step + 1) / warmup_iter
    # Constant
    elif step <= MAX_ITER - cooldown_iter:
        return 1.0

    # Cosine decay
    decay_ratio = (step - (MAX_ITER - cooldown_iter)) / cooldown_iter
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return FINAL_LR_RATIO + coeff * (1.0 - FINAL_LR_RATIO)


# Warmup Muon optimizer momentum to 0.95
def get_muon_momentum(step: int):
    warmup_steps = MAX_ITER * MUON_MOMENTUM_WARMUP_ITER_RATIO
    cooldown_steps = MAX_ITER * MUON_MOMENTUM_COOLDOWN_ITER_RATIO
    cooldown_start = MAX_ITER - cooldown_steps

    if step < warmup_steps:
        frac = step / warmup_steps
        momentum = MUON_MOMENTUM_MIN + frac * (MUON_MOMENTUM_MAX -
            MUON_MOMENTUM_MIN)
    elif step > cooldown_start:
        frac = (step - cooldown_start) / cooldown_steps
        momentum = MUON_MOMENTUM_MAX - frac * (MUON_MOMENTUM_MAX -
            MUON_MOMENTUM_MIN)
    else:
        momentum = MUON_MOMENTUM_MAX

    return momentum


# Sample prompts
sample_prompts = [
    'The capital of France is',
    'The chemical symbol of gold is',
    'If yesterday was Friday, then tomorrow will be',
    'The opposite of hot is',
    'The planets of the solar system are:',
    'My favorite color is',
    'If 5*x + 3 = 13, then x is',
]


# Tensor to store validation loss
eval_loss_record = torch.tensor([], dtype=torch.float32)


# Godspeed!
for step in range(MAX_ITER + 1):
    last_step = step == MAX_ITER

    # Evaluate validation loss
    if last_step or (step > 0 and step % EVAL_EVERY == 0):
        # All processes should perform the evaluation.
        model.eval()

        eval_loss = torch.tensor(0.0, device=device)
        with torch.no_grad(), autocast_ctx:
            for _ in range(EVAL_STEPS):
                for _ in range(GRAD_ACCUM_STEPS):
                    inputs, targets = next(eval_loader)
                    eval_loss += model(inputs, targets)

        eval_loss /= EVAL_STEPS * GRAD_ACCUM_STEPS

        if distributed:
            dist.all_reduce(eval_loss, op=dist.ReduceOp.AVG)

        # Record and report validation loss
        eval_loss_record = torch.cat(
            (eval_loss_record, eval_loss.cpu().view(1)),
            dim=0
        )
        print0(f'{step=} evaluation loss: {eval_loss.item():.4f}')

        model.train()

    # Sample some tokens once in a while
    if master_process and (last_step or (step > 0 and step % SAMPLE_EVERY == 0)):
        model.eval()

        with torch.no_grad(), autocast_ctx:
            for i, prompt in enumerate(sample_prompts):
                print0(f'Sample {i + 1}: {prompt}', end='')

                tokens = tokenizer.encode(prompt)
                y = compiled_model.generate(tokens, 32)

                for token in y:
                    print0(tokenizer.decode([token]), end='')
                print0(end='\n\n')

        model.train()

    # Save the model once in a while
    if master_process and (step > 0 and step % CHECKPOINT_EVERY == 0):
        save_model(orig_model, optims, eval_loss_record)

    # Single training step
    t0 = time.perf_counter()
    # Train loss for logging
    train_loss = 0.0
    for micro_step in range(GRAD_ACCUM_STEPS):
        inputs, targets = next(train_loader)
        with autocast_ctx:
            loss = model(inputs, targets)
        loss /= GRAD_ACCUM_STEPS
        train_loss += loss.item()

        if distributed:
            model.require_backward_grad_sync = micro_step == GRAD_ACCUM_STEPS - 1

        loss.backward()

    optim_update_params(
        get_lr_multiplier(step),
        get_muon_momentum(step)
    )

    optim_step()
    optim_zero_grad()

    t1 = time.perf_counter()
    dt = t1 - t0
    tps = BATCH_SIZE // dt

    print0(f'{step=}, {train_loss=:.4f}, took={dt * 1000:.4f}ms, tok/sec={tps:d}')

    # Save the model in final step
    if master_process and last_step:
        save_model(orig_model, optims, eval_loss_record, 'fibergpt_pretrain.bin')

    if step == 0:
        gc.collect()
        gc.freeze()
        # Nuke the GC to reduce overhead
        gc.disable()
    elif step % 2500 == 0:
        # Collect every 2500 steps
        gc.collect()


compute_cleanup()
