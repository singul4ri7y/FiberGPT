import torch
import torch.distributed as dist
import math
import time
import gc
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
NO_OF_BATCH_PER_DEVICE = 64  # Change this based on GPU VRAM
GRAD_ACCUM_STEPS = NO_OF_BATCH // NO_OF_BATCH_PER_DEVICE

# Training hyperparams
MAX_ITER = 10_000  # Roughly enough to go through the entire dataset
WARMUP_ITER_RATIO = 0.01
WARMDOWN_ITER_RATIO = 0.5
FINAL_LR_RATIO = 0.1

# Sample, eval and checkpoint
SAMPLE_EVERY = 500
EVAL_EVERY = 500
EVAL_STEPS = 25
CHECKPOINT_EVERY = 500


# Warn if FA3 is not available.
if not fa3_available:
    print0(
        'WARNING: Flash Attention 3 is not available, using SDPA fallback.'
        ' Training will be inefficient!'
    )


# Training states
autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None


# Compile the model
orig_model = FiberGPT(FiberGPTConfig()).to(device)
model = torch.compile(orig_model, dynamic=False, fullgraph=True)

# Broadcast parameters from rank 0 constructed model, so that all the processes
# share the same parameters. Only for distributed training.
if distributed:
    for param in orig_model.parameters():
        dist.broadcast(param.detach(), 0)


# Optimizers
optims = model.setup_optimizers()  # Use the default optimizer parameters
opt_params = lambda opt: [p for group in opt.param_groups for p in group['params']]
opt_params = {
    optim: opt_params(optim) for optim in optims
}

def optim_step(optim_futures):
    for optim in optims:
        # Wait for ALL REDUCE operation to complete
        if optim_futures is not None:
            torch.futures.collect_all(optim_futures[optim]).wait()
        optim.step()

def optim_zero_grad():
    for optim in optims:
        optim.zero_grad(set_to_none=True)

def optim_update_params(lrm: float, momentum: float, weight_decay: float):
    for optim in optims:
        for group in optim.param_groups:
            group['lr'] = group['initial_lr'] * lrm

            if isinstance(optim, (Muon, DistributedMuon)):
                group['momentum'] = momentum
                group['weight_decay'] = weight_decay


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
def get_lr_multiplier(it: int):
    warmup_iter = round(WARMUP_ITER_RATIO * MAX_ITER)
    warmdown_iter = round(WARMDOWN_ITER_RATIO * MAX_ITER)

    # Linear warmup
    if it < warmup_iter:
        return (it + 1) / warmup_iter
    # Constant
    elif it <= MAX_ITER - warmdown_iter:
        return 1.0

    # Cosine decay
    decay_ratio = (it - (MAX_ITER - warmdown_iter)) / warmdown_iter
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return FINAL_LR_RATIO + coeff * (1.0 - FINAL_LR_RATIO)


# Warmup Muon optimizer momentum for first 300 steps to 0.95
def get_muon_momentum(it: int):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# Weight decay scheduler for Muon. Start with 0.1 and reduce to 0.0.
def get_weight_decay(it: int):
    return 0.1 * (1 - it / MAX_ITER)


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
        if distributed:
            dist.barrier()
        model.eval()

        eval_loss = 0
        with torch.no_grad(), autocast_ctx:
            for _ in range(EVAL_STEPS):
                for _ in range(GRAD_ACCUM_STEPS):
                    inputs, targets = next(eval_loader)
                    eval_loss += model.forward(inputs, targets)

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
        if distributed:
            dist.barrier()

    # Sample some tokens once in a while
    if last_step or (step > 0 and step % SAMPLE_EVERY == 0):
        if master_process:
            model.eval()

            with torch.no_grad(), autocast_ctx:
                for i, prompt in enumerate(sample_prompts):
                    print0(f'Sample {i + 1}: {prompt}', end='')

                    tokens = tokenizer.encode(prompt)
                    y = model.generate(tokens, 32)

                    for token in y:
                        print0(tokenizer.decode([token]), end='')
                    print0(end='\n\n')

            model.train()

        if distributed:
            dist.barrier()

    # Save the model once in a while
    if step > 0 and step % CHECKPOINT_EVERY == 0:
        save_model(orig_model, optims, eval_loss_record)

    # Single training step
    t0 = time.perf_counter()
    # Train loss for logging
    train_loss = 0.0
    for _ in range(GRAD_ACCUM_STEPS):
        inputs, targets = next(train_loader)
        with autocast_ctx:
            loss = model.forward(inputs, targets)
        loss /= GRAD_ACCUM_STEPS
        train_loss += loss.item()
        loss.backward()

    optim_futures = None
    if distributed:
        optim_futures = {
            opt: [
                dist.all_reduce(
                    p.grad, op=dist.ReduceOp.AVG, async_op=True
                ).get_future()
                for p in params
            ]
            for opt, params in opt_params.items()
        }

    optim_update_params(
        get_lr_multiplier(step),
        get_muon_momentum(step),
        get_weight_decay(step)
    )

    optim_step(optim_futures)
    optim_zero_grad()

    sync()

    t1 = time.perf_counter()
    dt = t1 - t0
    tps = BATCH_SIZE // dt

    print0(f'{step=}, took={dt * 1000:.4f}ms, tok/sec={tps}')

    # Save the model in final step
    if last_step:
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
