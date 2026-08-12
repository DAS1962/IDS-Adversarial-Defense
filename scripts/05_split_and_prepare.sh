
#!/bin/bash
#SBATCH --job-name=split_prepare
#SBATCH --account=def-smoolak
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

module load python/3.11
source ~/ENV/bin/activate
cd ~/IDS-Adversarial-Defense

python scripts/05_split_and_prepare.py

echo ""
echo "=== Job termine a : $(date) ==="