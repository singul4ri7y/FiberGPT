# Simple dataloader inspired by modded-nanogpt

import torch
import numpy as np
import threading
import random
from pathlib import Path
from utils.tokenizer import tokenizer, _extended_special_tokens


def _load_data_shard(file: Path):
    # Read the header
    header = torch.from_file(
        str(file),
        shareded=False,
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

    return tokens


# Randomize data shards for later epochs
def _shuffled_files_cycle(files):
    while True:
        yield from files
        random.shuffle(files)


# Distributed Data Generator for pretraining
def DDGPretrain(
    filename_pattern: str,
    batch_size: int,
    rank: int, world_size: int
):
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0, ('Batch size should be multiple of '
        'world_size')

    local_batch_size = batch_size // world_size
    file_iter = _shuffled_files_cycle(files)
    tokens = _load_data_shard(next(file_iter))
    tokens_size = len(tokens)
    pos = 0

    while True:
        # If this shard is exhausted
        if pos + batch_size + 1 >= tokens_size:
            tokens = _load_data_shard(next(file_iter))
            tokens_size = len(tokens)

        buff = tokens[pos + rank * local_batch_size:][:local_batch_size+1]
        inputs = buff[:-1].to(
            device=f'cuda:{rank}', dtype=torch.int32, non_blocking=True
        )
        targets = buff[1:].to(
            device=f'cuda:{rank}', dtype=torch.int32, non_blocking=True
        )

        pos += batch_size
        yield inputs, targets

