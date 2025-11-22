#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test_out_%A_%a.out
#SBATCH --error=test_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --gpus=1                                 
#SBATCH --mail-type=END,FAIL         
#SBATCH --mem-per-cpu=8G      
#SBATCH --mail-user=jiapan@student.ethz.ch
#SBATCH --mail-type=END,FAIL         

# activate conda env
source /cluster/home/jiapan/miniforge3/etc/profile.d/conda.sh
conda activate gravestones

# run script
python /cluster/home/jiapan/Supervised_Research/database/check_types.py
