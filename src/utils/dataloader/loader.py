# Simple dataloader inspired by modded-nanogpt

import torch
import random
from pathlib import Path
from .dataset import DatasetCollator


def _load_data_shard(file: Path):
    # Read the header
    header = torch.from_file(
        str(file),
        shared=False,
        size=256,
        dtype=torch.int32
    )
    assert header[0] == 20240520, 'Magic number mismatch in data shard!'
    assert header[1] == 1, 'Unsupported shard version!'

    num_tokens = int(header[2])
    with file.open('rb', buffering=0) as f:
        # Pin memory to reduce overhead
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)

        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, ('Number of tokens read does not match'
            ' metadata')

    return tokens.to(torch.int32)


# Randomize data shards for later epochs
def _shuffled_files_cycle(files):
    while True:
        yield from files
        random.shuffle(files)


# Distributed Data Generator for pretraining
def DDGPretrain(
    filename_pattern: str,
    device_batch_length: int, context_length: int,
    rank: int, world_size: int
):
    assert device_batch_length % world_size == 0, ('Batch size should be '
        'multiple of world_size')

    files = sorted(Path.cwd().glob(filename_pattern))
    device_batch_size = device_batch_length * context_length
    batch_size = device_batch_size * world_size  # Total batch size in tokens
    file_iter = _shuffled_files_cycle(files)
    tokens = _load_data_shard(next(file_iter))
    tokens_size = len(tokens)
    pos = 0

    while True:
        # If this shard is exhausted
        if pos + batch_size + 1 >= tokens_size:
            tokens = _load_data_shard(next(file_iter))
            tokens_size = len(tokens)
            pos = 0

        buff = tokens[pos + rank * device_batch_size:][:device_batch_size+1]
        inputs = buff[:-1].to(
            device=f'cuda:{rank}', dtype=torch.int32, non_blocking=True
        ).view(device_batch_length, context_length)
        targets = buff[1:].to(
            device=f'cuda:{rank}', dtype=torch.int32, non_blocking=True
        ).view(device_batch_length, context_length)

        pos += batch_size
        yield inputs, targets


# Distributed Data Generator for chat SFT
def DDGFinetune(
    collator: DatasetCollator,
    device_batch_length: int, context_length: int,
    rank: int, world_size: int
):
    assert device_batch_length % world_size == 0, ('Batch size should be '
        'multiple of world_size')

    device_batch_size = device_batch_length * context_length
    batch_size = device_batch_size * world_size  # Total batch size in tokens
    tokens, masks = collator.get_only_shard()
    tokens_size = len(tokens)
    pos = 0

    while True:
        # Finetune dataset is a single data shard of roughly ~550M
        if pos + batch_size + 1 >= tokens_size:
            tokens, masks = collator.get_only_shard()
            tokens_size = len(tokens)
            pos = 0

        buff = tokens[pos + rank * device_batch_size:][:device_batch_size+1]
        # Buffer for masking
        mbuff = masks[pos + rank * device_batch_size + 1:][:device_batch_size]
        inputs = buff[:-1].to(
            device=f'cuda:{rank}', dtype=torch.int32, non_blocking=True
        ).view(device_batch_length, context_length)
        targets = buff[1:].to(
            device=f'cuda:{rank}', dtype=torch.int32, non_blocking=True
        ).view(device_batch_length, context_length)

        # Mask targets
        targets[
            ~mbuff.to(
                device=f'cuda:{rank}', dtype=torch.bool, non_blocking=True
            ).view(device_batch_length, context_length)
        ] = -69

        pos += batch_size
        yield inputs, targets

