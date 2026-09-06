#!/bin/bash
#SBATCH --job-name=balanced_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

set -o pipefail

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Node       : $(hostname)"
echo "Job ID     : $SLURM_JOB_ID"
echo "========================="
echo

module load python/3.11
source ~/ENV/bin/activate

cd ~/IDS-Adversarial-Defense

python -u scripts/15_balanced_evaluation.py
statut=$?

echo
echo "=== Job SLURM termine a : $(date) ==="
echo "Code de sortie Python : $statut"
exit $statut
