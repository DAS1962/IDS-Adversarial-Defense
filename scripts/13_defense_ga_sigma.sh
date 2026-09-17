#!/bin/bash
#SBATCH --job-name=ga_sigma
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:30:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

# Balayage du niveau de bruit de l'augmentation gaussienne.
# Usage : sbatch scripts/13_defense_ga_sigma.sh 0.05
# La valeur est passee au script Python, qui nomme ses sorties en consequence
# (defense_ga_sigma0p05_*.pkl) pour ne pas ecraser les autres executions.

set -o pipefail

SIGMA=$1
if [ -z "$SIGMA" ]; then
    echo "ERREUR : valeur de sigma manquante."
    echo "Usage : sbatch scripts/13_defense_ga_sigma.sh 0.05"
    exit 2
fi

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Node       : $(hostname)"
echo "Job ID     : $SLURM_JOB_ID"
echo "Sigma      : $SIGMA"
echo "========================="
echo

nvidia-smi --query-gpu=name,memory.total --format=csv
echo

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Adversarial-Defense

python -u scripts/13_defense_gaussian_augmentation.py --sigma "$SIGMA"
statut=$?

echo
echo "=== Job termine a : $(date) ==="
echo "Code de sortie Python : $statut"
exit $statut
