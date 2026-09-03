#!/bin/bash
# DESACTIVE : 09_generate_attacks_jsma_sample.py refuse maintenant de
# s'executer (voir l'en-tete du .py). Ne pas soumettre ce job : il
# consommerait une allocation cluster pour un script qui s'arrete tout de
# suite. Utiliser scripts/08_generate_attacks.sh a la place.
#SBATCH --job-name=jsma_cw_sample
#SBATCH --account=def-smoolak
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=15:00:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

set -eo pipefail

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Node       : $(hostname)"
echo "Job ID     : $SLURM_JOB_ID"
echo "========================="
echo ""

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Adversarial-Defense

python scripts/09_generate_attacks_jsma_sample.py

echo ""
echo "=== Job termine a : $(date) ==="
