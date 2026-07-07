#!/bin/bash
#SBATCH --job-name=criscross_pretrain
#SBATCH --output=logs/pretrain_%A_%a.out
#SBATCH --error=logs/pretrain_%A_%a.err
#SBATCH --partition=REPLACE_ME            # e.g. gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24                # DataModule uses num_workers=20
#SBATCH --mem=64G
#SBATCH --time=REPLACE_ME                 # e.g. 24:00:00
#SBATCH --array=0-REPLACE_ME              # one task per entry in the config JSON, e.g. 0-9

# Full production run: reads configs/artificial_AG_param_combinationsWTC.json
# and runs the config[SLURM_ARRAY_TASK_ID] entry. Because SLURM_ARRAY_TASK_ID
# is set here (via --array), pretrainArtificial.py takes the --config branch
# instead of the hardcoded test defaults.

set -euo pipefail

mkdir -p logs

# --- environment setup ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate REPLACE_ME_CONDA_ENV       # e.g. conda activate criscross

# --- run from repo root so `CRISCross` resolves as a module ---
cd "$(dirname "$0")/.."

CONFIG_PATH="configs/artificial_AG_param_combinationsWTC.json"

srun python -m CRISCross.pretrainArtificial --config "${CONFIG_PATH}"
