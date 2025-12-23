#!/bin/bash
#SBATCH --job-name=task_finetune
#SBATCH --output=finetune_%A_%a.out
#SBATCH --error=finetune_%A_%a.err
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=nvidia_geforce_rtx_4090:1

#SBATCH --mail-user=jiapan@student.ethz.ch
#SBATCH --mail-type=END,FAIL

# print GPU information
nvidia-smi

# activate conda env
source /cluster/home/jiapan/miniforge3/etc/profile.d/conda.sh
conda activate gravestones

# unbuffer python output
export PYTHONUNBUFFERED=1

# run scripts
python /cluster/home/jiapan/Supervised_Research/scripts/run_linear_probing.py
