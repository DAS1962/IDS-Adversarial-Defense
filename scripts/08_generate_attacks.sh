#!/bin/bash
#SBATCH --job-name=gen_attacks
#SBATCH --account=def-smoolak_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Node       : $SLURMD_NODENAME"
echo "Job ID     : $SLURM_JOB_ID"
echo "========================="
echo

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo

module load python/3.11
source ~/ENV/bin/activate

cd ~/IDS-Adversarial-Defense

python -u scripts/08_generate_attacks.py

echo
echo "=== Job SLURM termine ==="
echo "Date fin : $(date)"
