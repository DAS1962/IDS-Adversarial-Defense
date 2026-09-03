#!/bin/bash
# Temps alloue augmente de 3h a 12h : le script genere maintenant aussi le
# modele substitut (30 epochs), et configs/config.yaml est revenu aux
# parametres JSMA fideles a l'article (theta=0.1, gamma=1.0), plus lents que
# les theta=0.3/gamma=0.15 utilises precedemment. Le script imprime une
# estimation de duree avant de lancer JSMA en grandeur reelle (voir
# warn_if_slow dans 08_generate_attacks.py) : verifier le log au demarrage.
# Si l'estimation depasse largement ce qui reste alloue, annuler le job et
# repasser evaluation.scope a "sample" dans configs/config.yaml plutot que
# de laisser tourner a l'aveugle.
#SBATCH --job-name=gen_attacks
#SBATCH --account=def-smoolak_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

echo "=== Job SLURM demarre ==="
echo "Date debut : $(date)"
echo "Node       : $SLURMD_NODENAME"
echo "Job ID     : $SLURM_JOB_ID"
echo "========================="
echo

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo

module load python/3.11
source ~/ENV/bin/activate

cd ~/IDS-Adversarial-Defense

python -u scripts/08_generate_attacks.py

echo
echo "=== Job SLURM termine ==="
echo "Date fin : $(date)"
