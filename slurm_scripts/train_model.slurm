#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# SBATCH --gpus=nvidia_a100_80gb_pcie:1
# SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --gpus=nvidia_geforce_rtx_3090:1

#SBATCH --mail-user=jiapan@student.ethz.ch
#SBATCH --mail-type=END,FAIL

# print GPU information
nvidia-smi

# activate conda env
source /cluster/home/jiapan/miniforge3/etc/profile.d/conda.sh
conda activate gravestones

# unbuffer python output
export PYTHONUNBUFFERED=1

# run script
#python /cluster/home/jiapan/Supervised_Research/scripts/train.py mae
python /cluster/home/jiapan/Supervised_Research/scripts/train.py dino
