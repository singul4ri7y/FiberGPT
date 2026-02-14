# Common utilities

import os
import torch
import torch.distributed as dist
from utils.muon import Muon, DistributedMuon


def print0(*args,**kwargs):
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        print(*args, **kwargs)


def is_dist_requested() -> bool:
    ''' Are scripts launched via torchrun? '''

    return all(k in os.environ for k in [ 'RANK', 'LOCAL_RANK', 'WORLD_SIZE' ])


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    if is_dist_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        dist_rank = int(os.environ['RANK'])
        dist_local_rank = int(os.environ['LOCAL_RANK'])
        dist_world_size = int(os.environ['WORLD_SIZE'])
        return True, dist_rank, dist_local_rank, dist_world_size

    return False, 0, 0, 1


def compute_init(device_type='cuda'):
    assert device_type in ['cuda', 'mps', 'cpu'], 'Invalid device type!'
    if device_type == 'cuda':
        assert torch.cuda.is_available(), 'Your PyTorch is not CUDA configured!'
    if device_type == 'mps':
        assert torch.backends.mps.is_available(), 'Your PyTorch is not MPS configured!'

    # Reproducibility
    torch.manual_seed(42)
    if device_type == 'cuda':
        torch.cuda.manual_seed(42)

    # Precision (especially for float32)
    if device_type == 'cuda':
        torch.backends.fp32_precision = 'tf32'
        torch.set_float32_matmul_precision('medium')

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    distributed, rank, local_rank, world_size = get_dist_info()
    if distributed and device_type == 'cuda':
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)  # Make CUDA:<rank> default device
        dist.init_process_group(backend='nccl', device_id=device)

        # Wait for other distributed initialization to complete
        dist.barrier()
    else:
        device = torch.device(device_type)

    return distributed, rank, local_rank, world_size, device


def compute_cleanup():
    if is_dist_initialized():
        dist.destroy_process_group()


def save_model(model, optims, eval_loss_record, name='fibergpt_pretrain_save.bin'):
    adamw_state_dict, muon_state_dict = None, None
    for optim in optims:
        if isinstance(optim, (Muon, DistributedMuon)):
            muon_state_dict = optim.state_dict()
        else:
            adamw_state_dict = optim.state_dict()

    torch.save({
        'model': model.state_dict(),
        'optim_adam': adamw_state_dict,
        'optim_muon': muon_state_dict,
        'loss_list': eval_loss_record
    }, name)
