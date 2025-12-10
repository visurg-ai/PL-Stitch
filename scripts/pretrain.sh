#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0     # ← pin to GPU 0

python -m torch.distributed.run \
  --nproc_per_node=1 \
  --nnodes=1 \
  --master_port="${MASTER_PORT:-29502}" \
  main_pl.py \
      --arch vit_base \
      --output_dir gtea_lpw_vitbase16 \
      --data_path /workspace/lemon/surgical-datasets/gtea_train.lmdb \
      --load_from checkpoint.pth \
      --epochs 100 \
      --warmup_epochs 10 \
      --lambda_temporal 1.0 \
      --start_epoch_video 1 \
      --start_epoch_puzzle 3 \
      --batch_size_per_gpu 100 \
      --batch_size_temporal_per_gpu 20 \
      --saveckp_freq 40 \
      --lr 0.0001 \
      --lr_head 0.00035 \
      --local_crops_number 0 \
      --global_crops_scale 0.14 1 \
      --local_crops_scale 0.05 0.25 \
      --lambda_video 1.0 \
      --lambda_puzzle 0.4 \
      --momentum_teacher 0.998

