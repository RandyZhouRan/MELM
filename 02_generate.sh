#!/bin/bash

LOAD_BERT="xlm-roberta-base"
CKPT_DIR="./ckpt/ckpt.pt"
IN_DIR="./data/"

python generate.py \
  --seed=42 \
  --bsize=16 \
  --mu_ratio=0.5 \
  --sigma=1 \
  --o_mask_rate=0 \
  --k=5 \
  --sub_idx=-1 \
  --ckpt_dir=${CKPT_DIR} \
  --load_bert=${LOAD_BERT} \
  --in_dir=${IN_DIR}
