#!/bin/bash
#SBATCH --job-name=plot_results
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Job ID     : $SLURM_JOB_ID"
echo "========================="

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Adversarial-Defense

python -u scripts/07_plot_results.py

echo "=== Job termine a : $(date) ==="
