#!/bin/bash
#SBATCH --job-name=feature_selection
#SBATCH --account=def-smoolak
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

set -eo pipefail

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Node       : $(hostname)"
echo "Job ID     : $SLURM_JOB_ID"
echo "CPUs       : $SLURM_CPUS_PER_TASK"
echo "Memoire    : $SLURM_MEM_PER_NODE MB"
echo "========================="
echo ""

# Charger les modules necessaires
module load python/3.11

# Activer l'environnement virtuel
source ~/ENV/bin/activate

# Se placer dans le dossier du projet
cd ~/IDS-Adversarial-Defense

# Executer le script Python
python scripts/04_feature_selection.py

echo ""
echo "=== Job termine a : $(date) ==="
