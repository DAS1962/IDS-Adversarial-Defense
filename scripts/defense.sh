#!/bin/bash
#SBATCH --job-name=defense
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

# Usage : sbatch scripts/defense.sh 07_defense_ls
set -o pipefail
SCRIPT=$1
if [ -z "$SCRIPT" ]; then
    echo "Usage : sbatch scripts/defense.sh {07_defense_ls|08_defense_ga|09_defense_at|10_defense_dae}"
    exit 2
fi

echo "Debut : $(date)"
echo "Script : $SCRIPT"
echo

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Reproduction-Fidele

# shift retire le nom du script, "$@" relaie le reste
shift
python -u "scripts/${SCRIPT}.py" "$@"
statut=$?

echo
echo "Fin : $(date)"
echo "Code de sortie : $statut"
exit $statut
