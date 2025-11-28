#!/bin/bash
#SBATCH --job-name=gen_gravestones
#SBATCH --output=out_%A_%a.out
#SBATCH --error=err_%A_%a.err
#SBATCH --time=300:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=6G

#SBATCH --mail-user=jiapan@student.ethz.ch
#SBATCH --mail-type=END,FAIL

# activate conda env
source /cluster/home/jiapan/miniforge3/etc/profile.d/conda.sh
conda activate gravestones

# unbuffer python output
export PYTHONUNBUFFERED=1

# run script
python /cluster/home/jiapan/Supervised_Research/scripts/gen_new_dataset.py
