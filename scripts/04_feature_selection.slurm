
#!/bin/bash
#SBATCH --job-name=feature_selection
#SBATCH --account=def-smoolak
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

# Afficher les informations du job
echo "======================================"
echo "Job SLURM demarre"
echo "Date debut : $(date)"
echo "Node       : $(hostname)"
echo "Job ID     : $SLURM_JOB_ID"
echo "CPU count  : $SLURM_CPUS_PER_TASK"
echo "======================================"
echo ""

# Charger le module Python
module load python/3.11

# Activer l'environnement virtuel
source ~/ENV/bin/activate

# Aller dans le dossier du projet
cd ~/IDS-Adversarial-Defense

# Lancer le script Python
python scripts/04_feature_selection.py

# Afficher les infos de fin
echo ""
echo "======================================"
echo "Job termine a : $(date)"
echo "======================================"