#!/usr/bin/env bash

# Download pre-tokenized FineWeb 10B dataset
python data/fineweb10b.py

# Pretrain
torchrun --nproc-per-node=8 src/pretrain.py

# Chat SFT
torchrun --nproc-per-node=8 src/sft.py
