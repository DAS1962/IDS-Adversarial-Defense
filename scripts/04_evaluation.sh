#!/bin/bash
#SBATCH --job-name=eval_fidele
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

set -o pipefail
echo "Debut : $(date)"
echo

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Reproduction-Fidele

python -u scripts/04_evaluation.py
statut=$?

echo
echo "Fin : $(date)"
echo "Code de sortie : $statut"
exit $statut
