#!/bin/bash

# Print job information
echo "==========================================" >> gpu_out.txt
echo "Job started on: $(date)" >> gpu_out.txt
echo "Job ID: $SLURM_JOB_ID" >> gpu_out.txt
echo "Running on node: $SLURMD_NODENAME" >> gpu_out.txt
echo "Number of CPUs: $SLURM_CPUS_PER_TASK" >> gpu_out.txt
echo "Working directory: $PWD" >> gpu_out.txt
echo "==========================================" >> gpu_out.txt

# Print system information
echo "System information:" >> gpu_out.txt
echo "Hostname: $(hostname)" >> gpu_out.txt
echo "Operating System: $(uname -a)" >> gpu_out.txt
echo "CPU info: $(lscpu | grep 'Model name' | head -1)" >> gpu_out.txt
echo "Memory info: $(free -h | grep 'Mem:')" >> gpu_out.txt
echo "" >> gpu_out.txt


# RUN JOB
# conda init
# conda activate gravestones
srun --gpus=a100:1 /cluster/home/jiapan/miniforge3/envs/gravestones/bin/python /cluster/home/jiapan/Supervised_Research/dataset/print_gpu_info.py >> gpu_out.txt


# Job finished message
echo "" >> gpu_out.txt
echo "==========================================" >> gpu_out.txt
echo "Job completed on: $(date)" >> gpu_out.txt
echo "Total runtime: $SECONDS seconds" >> gpu_out.txt
echo "==========================================" >> gpu_out.txt