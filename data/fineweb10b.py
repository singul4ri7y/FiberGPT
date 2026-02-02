import os
import sys
from huggingface_hub import hf_hub_download

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

# Download training splits
for i in range(1, num_chunks + 1):
    get_file(f'fineweb_train_{i:06d}.bin')
