import torch
import torch.distributed as dist
import math
import time
import gc
from model import FiberGPT, FiberGPTConfig
from utils.common import compute_init, compute_cleanup, print0
from utils.attention import fa3_available
from utils.dataloader import DDGPretrain
from utils.muon import Muon, DistributedMuon


device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
# Are we going distribtued?
distributed, rank, local_rank, world_size, device = compute_init(device_type)


# HYPERPARAMETERS
CONTEXT_LENGTH = FiberGPTConfig.context_length
BATCH_SIZE = 1_048_576  # ~1M in tokens
NO_OF_BATCH = BATCH_SIZE // (world_size * CONTEXT_LENGTH)
NO_OF_DEVICE_BATCH = 32  # Change this based on GPU VRAM
GRAD_ACCUM_STEPS = NO_OF_BATCH // NO_OF_DEVICE_BATCH
# Training hyperparams
MAX_ITER = 10_000  # Roughly enough to go through the entire dataset
WARMUP_ITER_RATIO = 0.01
WARMDOWN_ITER_RATIO = 0.5
FINAL_LR_RATIO = 0.1


# Warn if FA3 is not available.
if not fa3_available:
    print0(
        'WARNING: Flash Attention 3 is not available, using SDPA fallback.'
        ' Training will be inefficient!'
    )


# Training states
autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16)
sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda:None


# Compile the model
orig_model = FiberGPT(FiberGPTConfig()).to(device)
model = torch.compile(orig_model, dynamic=False, fullgraph=True)

# Broadcast parameters from rank 0 cosntructed model, so that all the processes
# share the same parameters.
for param in orig_model.parameters():
    dist.broadcast(param.detach(), 0)


# Optimizers
optims = model.setup_optimizers()  # Use the default optimizer parameters
opt_params = lambda opt: [p for group in opt.param_groups for p in group['params']]
opt_params = {
    optim: opt_params(optim) for optim in optims
}

def optim_step(opt_futures):
    for optim in optims:
        # Wait for ALL REDUCE operation to complete
        torch.futures.collect_all(opt_futures[optim]).wait()
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
    BATCH_SIZE, CONTEXT_LENGTH, local_rank, world_size
)
val_loader = DDGPretrain(
    'data/fineweb10b/fineweb_val_*.bin',
    BATCH_SIZE, CONTEXT_LENGTH, local_rank, world_size
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
    decay_ratio = (MAX_ITER - warmdown_iter) / warmdown_iter
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


# Godspeed!
for step in range(MAX_ITER + 1):
    last_step = step == MAX_ITER

    # Single training step
    sync()
    t0 = time.perf_counter()
    inputs, targets = next(train_loader)
    # Train loss for logging
    train_loss = 0.0
    for _ in range(GRAD_ACCUM_STEPS):
        with autocast_ctx:
            loss = model(inputs, targets)
        loss /= GRAD_ACCUM_STEPS
        train_loss += loss.item()
        loss.backward()

    opt_futures = {
        opt: [
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True)
            for p in params
        ]
        for opt, params in opt_params.items()
    }

    optim_update_params(
        get_lr_multiplier(step),
        get_muon_momentum(step),
        get_weight_decay(step)
    )

    optim_step(opt_futures)
    optim_zero_grad()

    sync()

    t1 = time.perf_counter()

    if step == 0:
        gc.collect()
        gc.freeze()
        # Nuke the GC to reduce overhead
        gc.disable()
    elif step % 2500 == 0:
        # Collect every 2500 steps
        gc.collect()


compute_cleanup()
