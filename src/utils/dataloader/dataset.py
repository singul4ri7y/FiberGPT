# Everythin about SFT datasets

import os
import torch
import re
import random
import json
import multiprocessing as mp
import urllib.request as req
from typing import List, Tuple
from functools import partial
from datasets import load_dataset
from tqdm import tqdm
from filelock import FileLock
from utils.tokenizer import tokenize_conversation


# A large variety of 370k english words
WORDS_LIST_URL = 'https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt'
IDENTITY_CONVERSATIONS_URL = 'https://drive.usercontent.google.com/download?id=1oElUyQEpCw3KewczXOGMrLd6Kh4VC_bN&export=download'
IDENTITY_CONVERSATIONS_FILENAME = 'identity_conversations.txt'
# Seed offset for testing dataset.
TEST_RANDOM_SEED_OFFSET = 10_000
# Letters of the alphabet
LETTERS = "abcdefghijklmnopqrstuvwxyz"


USER_MSG_TEMPLATES = [
    'How many {letter} are in the word {word}',
    'How many {letter} are in {word}',
    'Count the number of {letter} in {word}',
    'How many times does {letter} appear in {word}',
    "What's the count of {letter} in {word}",
    'In the word {word}, how many {letter} are there',
    'How many letter {letter} are in the word {word}',
    'Count how many {letter} appear in {word}',
    'Tell me the number of {letter} in {word}',
    'How many occurrences of {letter} are in {word}',
    'Find the count of {letter} in {word}',
    'Can you count the {letter} letters in {word}',
    'What is the frequency of {letter} in {word}',
    'How many {letter}s are in {word}',
    "How many {letter}'s are in {word}",
    'Count all the {letter} in {word}',
    'How many times is {letter} in {word}',
    'Number of {letter} in {word}',
    'Total count of {letter} in {word}',
    'How many {letter} does {word} have',
    'How many {letter} does {word} contain',
    "What's the number of {letter} in {word}",
    '{word} has how many {letter}',
    'In {word}, count the {letter}',
    'How many {letter} appear in {word}',
    'Count the {letter} in {word}',
    'Give me the count of {letter} in {word}',
    'How many instances of {letter} in {word}',
    'Show me how many {letter} are in {word}',
    'Calculate the number of {letter} in {word}'
]


## HELPER FUNCTIONS ##

def _render_mc(question, letters, choices):
    '''
    Multiple choice questions rendering.

    According to Karpathy, smaller models prefers the choice letters after
    the question. And, letters should not have any leading spaces as they
    might be tokenized differently, e.g. ' A' is a different token than 'A'.
    '''

    query = f'Multiple Choice question: {question}\n'
    query += ''.join([
        f'- {choice}={letter}\n'
        for letter, choice in zip(letters, choices)
    ])
    query += '\nRespond only with the letter of the correct answer.'
    return query


def _download_file_with_lock(url, filename):
    '''
    Downloads a file with lock so that no concurrent download happen between
    the ranks.
    '''

    file_path = os.path.join(os.getcwd(), filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        if os.path.exists(file_path):  # recheck
            return file_path

        # Download the content as bytes
        print(f'Downloading file {url}')
        with req.urlopen(url) as response:
            content = response.read()  # In bytes

        # Write to local file
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

    return file_path


class Dataset:
    ''' Simple dataset base class. '''

    def __init__(self, dataset, start: int = 0, stop: int = None, step: int = 1):
        # Sanity check
        assert start >= 0, f'Start must be non-negative, got {start}'
        assert stop is None or stop >= start, (f'Stop should be greater than '
            f'or equal to start, got {stop} and {start}')
        assert step >= 1, f'Step must be strictly positive, got {step}'

        self.start = start
        self.stop = stop  # Could be None here
        self.step = step

        # Dataset
        self.ds = dataset
        self.length = len(dataset)

        # Apply start, stop and step in the dataset.
        if self.stop is not None:
            self.ds = self.ds.select(
                range(self.start, min(self.length, self.stop), self.step)
            )
        else:
            self.ds = self.ds.select(range(self.start, self.length, self.step))

    def __len__(self):
        return self.length

    def collate(self) -> torch.Tensor:
        raise NotImplementedError


class SmolTalk(Dataset):
    ''' HuggingFace SmolTalk-smol dataset, train 460K rows, test 24K rows. '''

    def __init__(self, split: str, num_workers: int = None, **kwargs):
        assert split in ('train', 'test'), 'Invalid split'

        # Using all cores will certainly make us data bound. On top of that
        # there is poor cache locality overall.
        num_allowed_worker = min(16, mp.cpu_count())
        self.num_workers = (num_allowed_worker if num_workers is None else
            num_workers)

        ds = load_dataset(
            'HuggingFaceTB/smol-smoltalk',
            split=split
        ).shuffle(seed=42)
        super().__init__(ds, **kwargs)

    @staticmethod
    def _tokenize_chunk(
        args: Tuple,
        dataset
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end, worker_id = args

        # Holds torch tensors
        tokens, masks = [], []

        # Batch access
        ds_shard = dataset[start:end]
        for message in tqdm(
            ds_shard['messages'],
            desc=f'Tokenizing SmolTalk, worker id={worker_id}',
            position=worker_id,
            leave=True,
            total=end - start
        ):
            row_tokens, row_masks = tokenize_conversation(message)
            tokens.append(torch.tensor(row_tokens, dtype=torch.uint16))
            masks.append(torch.tensor(row_masks, dtype=torch.bool))

        return torch.cat(tokens, dim=0), torch.cat(masks, dim=0)

    def collate(self) -> torch.Tensor:
        length = len(self)
        load = length // self.num_workers
        # Compute chunk boundaries
        boundaries = [i * load for i in range(self.num_workers)] + [length]

        args = [
            (boundaries[i], boundaries[i + 1], i)
            for i in range(self.num_workers)
        ]

        # Dataset argument is fixed
        worker_fn = partial(SmolTalk._tokenize_chunk, dataset=self.ds)

        with mp.Pool(processes=self.num_workers) as pool:
            results = pool.map(worker_fn, args)

        all_tokens = torch.cat([ r[0] for r in results ], dim=0)
        all_masks  = torch.cat([ r[1] for r in results ], dim=0)

        return all_tokens, all_masks


class MMLU(Dataset):
    ''' MMLU multiple-choice dataset. '''

    letters = ('A', 'B', 'C', 'D')

    def __init__(
        self,
        subset: str,
        split: str,
        num_workers: int = None,
        **kwargs
    ):
        assert subset in ['all', 'auxiliary_train'], (
            f'subset {subset} must be all or auxiliary_train'
        )
        assert split in ['train', 'validation', 'dev', 'test'], (
            f'Split {split} must be train/validation/dev/test'
        )

        ds = load_dataset('cais/mmlu', subset, split=split).shuffle(seed=42)
        if subset == 'auxiliary_train':
            assert split == 'train', 'auxiliary_train must be split into train'

            # For whatever reason auxiliary_train contents are inside a `train`
            # Wrapper
            ds = ds.map(
                lambda row: row['train'],
                remove_columns=['train']
            )

        super().__init__(ds, **kwargs)

        # Use multiprocessing for auxiliary_train (100K rows), not for others
        self.use_mp = subset == 'auxiliary_train' or self.length > 30_000
        if self.use_mp:
            num_allowed_workers = min(4, mp.cpu_count())
            self.num_workers = (
                num_allowed_workers if num_workers is None else num_workers
            )

        # For caching
        self.cache = None

    @staticmethod
    def _tokenize_chunk(
        args: Tuple, dataset, letters
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end, worker_id = args

        tokens, masks = [], []
        ds_shard = dataset[start:end]
        for question, answer, choices in tqdm(
            zip(ds_shard['question'], ds_shard['answer'], ds_shard['choices']),
            desc=f'Tokenizing MMLU, worker id={worker_id}',
            position=worker_id,
            leave=True,
            total=end - start,
        ):
            user_msg = _render_mc(question, letters, choices)
            assistant_msg = f'The answer would be - {letters[answer]}'

            messages = [
                { 'role': 'user', 'content': user_msg },
                { 'role': 'assistant', 'content': assistant_msg },
            ]

            row_tokens, row_masks = tokenize_conversation(messages)
            tokens.append(torch.tensor(row_tokens, dtype=torch.uint16))
            masks.append(torch.tensor(row_masks, dtype=torch.bool))

        return torch.cat(tokens, dim=0), torch.cat(masks, dim=0)

    def collate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache is not None:
            return self.cache

        if self.use_mp:
            # Use multiprocessing for large no. of rows
            length = len(self)
            load = length // self.num_workers
            boundaries = [i * load for i in range(self.num_workers)] + [length]
            args = [
                (boundaries[i], boundaries[i + 1], i)
                for i in range(self.num_workers)
            ]

            worker_fn = partial(
                MMLU._tokenize_chunk, dataset=self.ds, letters=self.letters
            )

            with mp.Pool(processes=self.num_workers) as pool:
                results = pool.map(worker_fn, args)

            all_tokens = torch.cat([r[0] for r in results], dim=0)
            all_masks = torch.cat([r[1] for r in results], dim=0)
        else:
            # Single-threaded for small no. or rows
            all_tokens, all_masks = MMLU._tokenize_chunk(
                (0, len(self), 0),
                dataset=self.ds, letters=self.letters
            )

        # We might do multiple epochs on MMLU, so cache to avoid re-tokenization.
        self.cache = (all_tokens, all_masks)
        return all_tokens, all_masks


class GSM8K(Dataset):
    '''
    OpenAI GSM8K dataset with grade-school maths.

    Mathematical expressions are wrapped with << >>. So, we need to extract
    the mathematical expression and use our <|python_start|> and <|python_end|>
    tokens as wrappers.
    '''

    def __init__(self, subset: str, split: str, **kwargs):
        assert subset in ['main', 'socratic'], (
            'GSM8K subset must be main/socratic'
        )
        assert split in ['train', 'test'], 'GSM8K split must be train/test'

        ds = load_dataset(
            'openai/gsm8k', subset, split=split
        ).shuffle(seed=42)
        super().__init__(ds, **kwargs)

        # GSM8K dataset might be used multiple times, so cache the collated
        # tokens.
        self.cache = None

    def collate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache is not None:
            return self.cache

        tokens, masks = [], []
        for question, answer in tqdm(
            zip(self.ds['question'], self.ds['answer']),
            desc='Tokenizing GSM8K'
        ):
            # Create conversation with proper assistant parts
            assistant_parts = []
            # Parts will be split in such manner:
            # ['text', '<<mathematical expr>>', 'text', '<<mathematical expr>>']
            parts = re.split(f'(<<[^>]+>>)', answer)
            for part in parts:
                # Expression
                if part.startswith('<<') and part.endswith('>>'):
                    # Calculator tool call
                    part = part[2:-2]  # Remove << >>
                    # Each of the expression consists of an answer
                    if '=' in part:
                        expr, result = part.split('=')
                    else:
                        expr, result = part, ''

                    # Calculator call part
                    assistant_parts.append({ 'type': 'python', 'text': expr })
                    # Also append the result
                    assistant_parts.append({ 'type': 'output', 'text': result })
                else:
                    assistant_parts.append({ 'type': 'text', 'text': part })

            messages = [
                { 'role': 'user', 'content': question },
                { 'role': 'assistant', 'content': assistant_parts }
            ]

            row_tokens, row_masks = tokenize_conversation(messages)
            tokens.append(torch.tensor(row_tokens, dtype=torch.uint16))
            masks.append(torch.tensor(row_masks, dtype=torch.bool))

        all_tokens = torch.cat(tokens, dim=0)
        all_masks = torch.cat(masks, dim=0)

        self.cache = (all_tokens, all_masks)
        return all_tokens, all_masks


class SpellingBee(Dataset):
    ''' Count occurrences of letter occurrences in the word. '''

    def __init__(self, size: int, split: str, num_workers: int = None):
        assert split in ('train', 'test')
        self.size = size
        self.split = split

        filename = WORDS_LIST_URL.split('/')[-1]
        path = _download_file_with_lock(WORDS_LIST_URL, filename)
        with open(path, encoding='utf-8') as f:
            self.words = [l.strip() for l in f]
        self.length = len(self.words)

        # Use multiprocessing for large sized data
        self.use_mp = size >= 30_000
        if self.use_mp:
            num_allowed_worker = min(8, mp.cpu_count())
            self.num_workers = (
                num_allowed_worker if num_workers is None else num_workers
            )

    @staticmethod
    def _tokenize_chunk(
        args: Tuple, words, split
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end, worker_id = args
        rng = random.Random(worker_id)

        tokens, masks = [], []
        for i in tqdm(
            range(start, end),
            desc=f'Tokenizing SpellingBee, worker id={worker_id}',
            position=worker_id,
            leave=True,
            total=end - start
        ):
            seed = worker_id + (i if split == 'train' else
                TEST_RANDOM_SEED_OFFSET + i)
            rng.seed(seed)

            word = rng.choice(words)
            # Pick a letter from the word (90%) or any random letter which
            # might not exist in the word (10%).
            letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)
            count = word.count(letter)

            # User message
            template = rng.choice(USER_MSG_TEMPLATES)
            if rng.random() < 0.3:
                template = template.lower()
            letter_quote = rng.choice(['', "'", '"'])
            word_quote = rng.choice(['', "'", '"'])
            user_msg = template.format(
                letter=f'{letter_quote}{letter}{letter_quote}',
                word=f'{word_quote}{word}{word_quote}')

            # 50% of the time use '?'.
            if rng.random() < 0.5:
                user_msg += '?'

            # Assistant response - as parts (text + python calls)
            assistant_parts = []
            word_letters = ' - '.join(list(word))
            lines = [
                f"I am asked to find the number of '{letter}' in the word '{word}'."
                f' Let me try a manual approach first.\n\nFirst spell the word out:\n'
                f"{word}: {word_letters}\n\nThen count the occurrences of '{letter}':\n"
            ]

            running = 0
            for j, ch in enumerate(word, 1):
                if ch == letter:
                    running += 1
                    lines.append(f'{j}: {ch} hit! count={running}\n')
                else:
                    lines.append(f'{j}: {ch}\n')

            lines.append(f'\nThis gives us {running}.')

            assistant_parts.append({ 'type': 'text', 'text': ''.join(lines) })
            # Python verification
            assistant_parts.append({
                'type': 'text',
                'text': '\n\nLet me double check this using Python.\n\n'
            })
            python_expr = f"'{word}'.count('{letter}')"
            assistant_parts.append({ 'type': 'python', 'text': python_expr })
            # Python output
            assistant_parts.append({ 'type': 'output', 'text': str(count) })
            # Final answer
            assistant_parts.append({
                'type': 'text',
                'text': f'Python gives us {count}. So my final answer is: {count}'
            })

            messages = [
                { 'role': 'user', 'content': user_msg },
                { 'role': 'assistant', 'content': assistant_parts },
            ]

            row_tokens, row_masks = tokenize_conversation(messages)
            tokens.append(torch.tensor(row_tokens, dtype=torch.uint16))
            masks.append(torch.tensor(row_masks, dtype=torch.bool))

        return torch.cat(tokens, dim=0), torch.cat(masks, dim=0)

    def collate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_mp:
            length = self.size  # No of words
            load = length // self.num_workers
            boundaries = [i * load for i in range(self.num_workers)] + [length]
            args = [
                (boundaries[i], boundaries[i + 1], i)
                for i in range(self.num_workers)
            ]

            worker_fn = partial(
                SpellingBee._tokenize_chunk, words=self.words, split=self.split
            )

            with mp.Pool(processes=self.num_workers) as pool:
                results = pool.map(worker_fn, args)

            all_tokens = torch.cat([r[0] for r in results], dim=0)
            all_masks = torch.cat([r[1] for r in results], dim=0)
        else:
            # Single-threaded for small no. of words
            all_tokens, all_masks = SpellingBee._tokenize_chunk(
                (0, self.size, 0),
                words=self.words, split=self.split
            )

        return all_tokens, all_masks


class SimpleSpelling(Dataset):
    ''' Spell the word `Fiber`. '''

    def __init__(self, size: int, split: str, num_workers: int = None):
        assert split in ('train', 'test')
        self.size = size
        self.split = split

        filename = WORDS_LIST_URL.split('/')[-1]
        path = _download_file_with_lock(WORDS_LIST_URL, filename)
        with open(path, encoding='utf-8') as f:
            self.words = [l.strip() for l in f]
        self.length = len(self.words)

        # Use multiprocessing for large sized data
        self.use_mp = size >= 30_000
        if self.use_mp:
            num_allowed_worker = min(8, mp.cpu_count())
            self.num_workers = (
                num_allowed_worker if num_workers is None else num_workers
            )

    @staticmethod
    def _tokenize_chunk(
        args: Tuple, words, split
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start, end, worker_id = args
        rng = random.Random(worker_id)

        tokens, masks = [], []
        for i in tqdm(
            range(start, end),
            desc=f'Tokenizing SimpleSpelling, worker id={worker_id}',
            position=worker_id,
            leave=True,
            total=end - start
        ):
            seed = worker_id + (i if split == 'train' else
                TEST_RANDOM_SEED_OFFSET + i)
            rng.seed(seed)

            word = rng.choice(words)
            word_letters = ' - '.join(list(word))

            messages = [
                { 'role': 'user', 'content': f'Spell the word: {word}' },
                {
                    'role': 'assistant',
                    'content': f'Spelling of {word}: {word_letters}'
                }
            ]

            row_tokens, row_masks = tokenize_conversation(messages)
            tokens.append(torch.tensor(row_tokens, dtype=torch.uint16))
            masks.append(torch.tensor(row_masks, dtype=torch.bool))

        return torch.cat(tokens, dim=0), torch.cat(masks, dim=0)

    def collate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_mp:
            length = self.size  # No of words
            load = length // self.num_workers
            boundaries = [i * load for i in range(self.num_workers)] + [length]
            args = [
                (boundaries[i], boundaries[i + 1], i)
                for i in range(self.num_workers)
            ]

            worker_fn = partial(
                SimpleSpelling._tokenize_chunk, words=self.words, split=self.split
            )

            with mp.Pool(processes=self.num_workers) as pool:
                results = pool.map(worker_fn, args)

            all_tokens = torch.cat([r[0] for r in results], dim=0)
            all_masks = torch.cat([r[1] for r in results], dim=0)
        else:
            # Single-threaded for small no. of words
            all_tokens, all_masks = SimpleSpelling._tokenize_chunk(
                (0, len(self), 0),
                words=self.size, split=self.split
            )

        return all_tokens, all_masks


class CustomJSON(Dataset):
    ''' Identity conversations dataset to inflict personality on FiberGPT. '''

    def __init__(self):
        # Download if file does not exist
        if not os.path.exists(IDENTITY_CONVERSATIONS_FILENAME):
            _download_file_with_lock(
                IDENTITY_CONVERSATIONS_URL,
                IDENTITY_CONVERSATIONS_FILENAME
            )

        self.conversations = []
        with open(IDENTITY_CONVERSATIONS_FILENAME, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                messages = json.loads(line)

                # Perform some sanity checks
                assert isinstance(messages, list), (
                    f'Expected list of messages, got {type(messages)}'
                )
                assert len(messages) >= 2, (
                    f'Conversation must have at least 2 messages, got {len(messages)}'
                )
                # Validate message structure
                for i, message in enumerate(messages):
                    assert 'role' in message, "The 'role' field is required!"
                    assert 'content' in message, "The 'content' field is required"
                    exp_role = 'user' if i % 2 == 0 else 'assistant'
                    assert message['role'] == exp_role, (f'Message has role '
                        f'{message['role']}, expected {exp_role}')
                    assert isinstance(message['content'], str)

                self.conversations.append(messages)
        self.length = len(self.conversations)

        # Tokenized cache
        self.cache = None

    def collate(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache is not None:
            return self.cache

        # Single threaded is enough as we only have ~1K rows
        tokens, masks = [], []
        for messages in tqdm(self.conversations, desc='Tokenizing CustomJSON'):
            row_tokens, row_masks = tokenize_conversation(messages)
            tokens.append(torch.tensor(row_tokens, dtype=torch.uint16))
            masks.append(torch.tensor(row_masks, dtype=torch.bool))

        all_tokens = torch.cat(tokens, dim=0)
        all_masks = torch.cat(masks, dim=0)

        # Cache tokens and masks
        self.cache = (all_tokens, all_masks)
        return all_tokens, all_masks


class DatasetCollator:
    ''' A DatasetCollator mixing the given datasets. '''

    def __len__(self) -> int:
        raise NotImplementedError

    def get_only_shard(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# NOTE: We can get away with a single shard system here, as the finetune
# datasets are quite small in comparison and we have gigabytes of RAM to spare.
class DatasetCollatorRandom(DatasetCollator):
    ''' DatasetCollator which randomizes the dataset in every epoch. '''

    def __init__(self, datasets: List[Dataset], seed: int = 42):
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

        self.collated = []
        for ds in datasets:
            self.collated.append(ds.collate())

    def __len__(self) -> int:
        length = 0
        for ds in self.collated:
            length += len(ds[0])

        return length

    def get_only_shard(self) -> Tuple[torch.Tensor, torch.Tensor]:
        perm = torch.randperm(len(self.collated), generator=self.rng)
        tokens = torch.cat([self.collated[i][0] for i in perm], dim=0)
        masks  = torch.cat([self.collated[i][1] for i in perm], dim=0)

        return tokens, masks


class DatasetCollatorSequential(DatasetCollator):
    ''' DatasetCollator with sequential mixing. '''

    def __init__(self, datasets: list[Dataset]):
        all_tokens, all_masks = [], []
        for ds in datasets:
            t, m = ds.collate()
            all_tokens.append(t)
            all_masks.append(m)

        self.tokens = torch.cat(all_tokens, dim=0)
        self.masks  = torch.cat(all_masks,  dim=0)


    def __len__(self) -> int:
        return len(self.tokens)

    def get_only_shard(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.tokens, self.masks
