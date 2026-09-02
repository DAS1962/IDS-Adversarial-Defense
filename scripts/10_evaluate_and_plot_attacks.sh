#!/bin/bash
#SBATCH --job-name=eval_plot_attacks
#SBATCH --account=def-smoolak
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Node       : $SLURMD_NODENAME"
echo "Job ID     : $SLURM_JOB_ID"
echo "========================="
echo

module load python/3.11
source ~/ENV/bin/activate

cd ~/IDS-Adversarial-Defense

python -u scripts/10_evaluate_and_plot_attacks.py

echo
echo "=== Job SLURM termine ==="
echo "Date fin : $(date)"
