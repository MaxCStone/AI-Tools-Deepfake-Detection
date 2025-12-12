#!/bin/bash
#SBATCH --job-name=training-model
#SBATCH --output=logs/MODEL_RUN_%j.log
#SBATCH --error=logs/MODEL_RUN_%j.err
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
 
uv run cnn/model.py