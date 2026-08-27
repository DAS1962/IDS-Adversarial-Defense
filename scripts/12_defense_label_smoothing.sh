
#!/bin/bash
#SBATCH --job-name=defense_ls
#SBATCH --account=def-smoolak_gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=03:00:00
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

python scripts/12_defense_label_smoothing.py

echo ""
echo "=== Job termine a : $(date) ==="