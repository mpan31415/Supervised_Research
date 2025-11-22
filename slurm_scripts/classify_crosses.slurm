#!/bin/bash
#SBATCH --job-name=classify_crosses
#SBATCH --output=classify_crosses_out%A_%a.out
#SBATCH --error=classify_crosses_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                   
#SBATCH --gpus=1                                 
#SBATCH --mail-type=END,FAIL         
#SBATCH --mem-per-cpu=8G      
#SBATCH --mail-user=jiapan@student.ethz.ch
#SBATCH --mail-type=END,FAIL         



module load stack/2024-06  gcc/12.2.0
module load python_cuda/3.11.6            

source gravestonesvenv/bin/activate

python classify_crosses_tar.py --checkpoint_path="b4_finetuned_15_model.pth" --tar_path="/path/to/some/shard/template-{000001..000119}.tar" --output_path="/path/to/output/"