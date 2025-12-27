#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python train_seal.py \
    --dataset_name 'cub' \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 200 \
    --num_workers 8 \
    --use_ssb_splits \
    --sup_weight 0.35 \
    --weight_decay 5e-5 \
    --transform 'imagenet' \
    --lr 0.1 \
    --eval_funcs 'v2' \
    --warmup_teacher_temp 0.07 \
    --teacher_temp 0.04 \
    --warmup_teacher_temp_epochs 30 \
    --memax_weight 1 \
    --exp_name cub_seal \
    --kl_temp 1.5 \
    --update_thd 0 \
    --memax_weight_1 0 \
    --memax_weight_2 0 \
    --unsupervised_smoothing 1.0 \
    --backbone_lr 0.05 \
    --model_name vit_dino_v2

