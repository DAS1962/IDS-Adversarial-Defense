#!/bin/bash
#SBATCH --job-name=baseline_varlr
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

set -o pipefail
echo "Debut : $(date)"
echo "Node  : $(hostname)"
echo

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Reproduction-Fidele

# "$@" relaie les arguments passes a sbatch vers le script Python
python -u scripts/02_baseline.py --variante-lr "$@"
statut=$?

echo
echo "Fin : $(date)"
echo "Code de sortie : $statut"
exit $statut
