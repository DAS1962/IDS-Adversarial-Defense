#!/bin/bash
#SBATCH --job-name=plot_defenses
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

set -o pipefail

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Job ID     : $SLURM_JOB_ID"
echo "========================="
echo

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Adversarial-Defense

python -u scripts/17_plot_defenses.py
statut=$?

echo
echo "=== Job termine a : $(date) ==="
echo "Code de sortie Python : $statut"
exit $statut
