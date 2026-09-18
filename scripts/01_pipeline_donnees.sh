#!/bin/bash
#SBATCH --job-name=pipeline_donnees
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

set -o pipefail
echo "Debut : $(date)"
echo "Node  : $(hostname)"
echo

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Reproduction-Fidele

python -u scripts/01_pipeline_donnees.py
statut=$?

echo
echo "Fin : $(date)"
echo "Code de sortie : $statut"
exit $statut
