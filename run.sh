#!/usr/bin/env bash

# Install flash attention 3 (for Hopper series GPUs)
echo 'Installing FA3...'
# CUDA 13
uv pip install https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.02.17-06dc5e7/flash_attn_3-3.0.0+20260217.cu130torch2100cxx11abitrue.fec3a6-cp39-abi3-linux_x86_64.whl
# CUDA 12.9
# uv pip install https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.02.17-06dc5e7/flash_attn_3-3.0.0+20260217.cu129torch2100cxx11abitrue.fec3a6-cp39-abi3-linux_x86_64.whl
# CUDA 12.8
# uv pip install https://github.com/windreamer/flash-attention3-wheels/releases/download/2026.02.17-06dc5e7/flash_attn_3-3.0.0+20260216.cu128torch2100cxx11abitrue.fec3a6-cp39-abi3-linux_x86_64.whl


# Download pre-tokenized FineWeb 10B dataset
python data/fineweb10b.py

# Pretrain
torchrun --nproc-per-node=8 src/pretrain.py

# Chat SFT
torchrun --nproc-per-node=8 src/sft.py
