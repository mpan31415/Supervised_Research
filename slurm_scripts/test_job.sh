#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --output=test_job_%j.out
#SBATCH --error=test_job_%j.err

# Print job information
echo "=========================================="
echo "Job started on: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Working directory: $PWD"
echo "=========================================="

# Print system information
echo "System information:"
echo "Hostname: $(hostname)"
echo "Operating System: $(uname -a)"
echo "CPU info: $(lscpu | grep 'Model name' | head -1)"
echo "Memory info: $(free -h | grep 'Mem:')"
echo ""

# Do some simple calculations
echo "Performing calculations..."
echo "Computing squares of numbers 1-10:"
for i in {1..10}; do
    square=$((i * i))
    echo "$i squared = $square"
done

# Create some output files
echo "Creating output files..."
echo "Hello from Euler!" > hello.txt
echo "Job completed successfully" > status.txt

# List files in current directory
echo ""
echo "Files created:"
ls -la *.txt

echo ""
echo "=========================================="
echo "Job completed on: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "=========================================="