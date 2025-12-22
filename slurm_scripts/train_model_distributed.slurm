#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=train_%A_%a.out
#SBATCH --error=train_%A_%a.err
#SBATCH --time=120:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=nvidia_geforce_rtx_4090:8

# SBATCH --gpus=nvidia_a100_80gb_pcie:4
# SBATCH --gpus=nvidia_geforce_rtx_3090:4

#SBATCH --mail-user=jiapan@student.ethz.ch
#SBATCH --mail-type=END,FAIL

##############################
# GPU INFO
##############################
nvidia-smi

##############################
# ACTIVATE ENV
##############################
source /cluster/home/jiapan/miniforge3/etc/profile.d/conda.sh
conda activate gravestones

##############################
# DEBUG + SPEED
##############################
export PYTHONUNBUFFERED=1
# export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1

##############################
# LAUNCH DDP TRAINING
##############################
torchrun --nproc_per_node=8 \
  /cluster/home/jiapan/Supervised_Research/scripts/train_distributed.py \
  /cluster/home/jiapan/Supervised_Research/scripts/config/train.yaml
