# Everythin about SFT datasets

import torch
import multiprocessing as mp
from typing import Tuple
from pathlib import Path
from datasets import load_dataset
from utils.tokenizer import tokenize_conversation


class Dataset:
    ''' Simple dataset base class. '''

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index: int):
        raise NotImplementedError

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def collate(self) -> torch.Tensor:
        raise NotImplementedError


class SmolTalk(Dataset):
    ''' HuggingFace SmolTalk-smol dataset, train 460K rows, test 24K rows. '''

    def __init__(self, split: str, num_workers: int):
        assert split in ('train', 'test'), 'Invalid split'


        self.num_workers = mp.cpu_count() if num_workers is None else num_workers
        self.ds = load_dataset(
            'HuggingFaceTB/smol-smoltalk',
            split=split
        ).shuffle(seed=42)
        self.length = len(self.ds)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.ds[index]['messages']

    # Self instance is picklable
    def _tokenize_chunk(
        self,
        indices: range
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Holds torch tensors
        tokens, masks = [], []

        for i in indices:
            row_tokens, row_masks = tokenize_conversation(self[i])
            tokens.append(torch.tensor(row_tokens, dtype=torch.uint16))
            masks.append(torch.tensor(row_masks, dtype=torch.bool))

        return torch.cat(tokens, dim=0), torch.cat(masks, dim=0)

    def collate(self) -> torch.Tensor:
        length = len(self)
        # Uneven load, most likely case
        if length % self.num_workers != 0:
            load = length // self.num_workers
            remain = length % self.num_workers

            # Construct sequential chunks
            chunks = []
            for i in range(self.num_workers):
                r = range(i * load, (i + 1) * load + (remain != 0))
                if remain != 0:
                    remain -= 1

                chunks.append(r)
        # Even load, unlikely, just to be bullet proof
        else:
            load = length // self.num_workers
            chunks = [
                range(i * load, (i + 1) * load)
                for i in range(self.num_workers)
            ]

        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(self._tokenize_chunk, chunks)

        # Collate the results
        all_tokens = torch.cat([ r[0] for r in results ], dim=0)
        all_masks = torch.cat([ r[1] for r in results ], dim=0)

        return all_tokens, all_masks


class DatasetCollatorRandom:
    pass


class DatasetCollatorSequential:
    pass
