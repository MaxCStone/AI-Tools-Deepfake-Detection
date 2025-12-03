#!/bin/bash
#SBATCH --job-name=training-model
#SBATCH --output=logs/HPT_RUN_%j.log
#SBATCH --error=logs/HPT_RUN_%j.err
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
 
uv run model.py \
    --data ../data/Dataset \
    --batch_size 32 \
    --epochs 2 \
    --main_dir . \
    --augment_data true \
    --fine_tune false