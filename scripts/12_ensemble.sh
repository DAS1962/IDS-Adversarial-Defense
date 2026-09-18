#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

set -o pipefail
echo "Debut : $(date)"
echo

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Reproduction-Fidele

python -u scripts/12_ensemble.py
statut=$?

echo
echo "Fin : $(date)"
echo "Code de sortie : $statut"
exit $statut
