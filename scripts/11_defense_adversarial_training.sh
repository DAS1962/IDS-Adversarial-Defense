#!/bin/bash
#SBATCH --job-name=defense_at
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:30:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

set -o pipefail

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Node       : $(hostname)"
echo "Job ID     : $SLURM_JOB_ID"
echo "========================="
echo

nvidia-smi --query-gpu=name,memory.total --format=csv
echo

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Adversarial-Defense

python -u scripts/11_defense_adversarial_training.py
statut=$?

echo
echo "=== Job termine a : $(date) ==="
echo "Code de sortie Python : $statut"
exit $statut
