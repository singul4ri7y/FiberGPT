import os
import sys
from huggingface_hub import hf_hub_download
import multiprocessing as mp

# Disable progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'fineweb10b')


# Huge shoutout to KJJ (https://huggingface.co/kjj0) for already GPT2 tokenized
# dataset. Wait, is he Keller Jordan?
def get_file(name):
    if not os.path.exists(os.path.join(DATA_DIR, name)):
        hf_hub_download(
            repo_id='kjj0/fineweb10B-gpt2',
            filename=name,
            repo_type='dataset',
            local_dir=DATA_DIR
        )

# Download validation split
get_file(f'fineweb_val_{0:06d}.bin')

num_chunks = 103
if len(sys.argv) >= 2:
    num_chunks = int(sys.argv[1])

# For parallel downloads
NUM_WORKERS = min(num_chunks, mp.cpu_count())

# Download training splits parallely
list_of_index_to_download = list(range(1, num_chunks + 1))
with mp.Pool(processes=NUM_WORKERS) as pool:
    results = pool.map(
        get_file,
        map(
            lambda i: f'fineweb_train_{i:06d}.bin',
            list_of_index_to_download
        )
    )
