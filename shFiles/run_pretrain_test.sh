#!/bin/bash
#SBATCH --job-name=criscross_pretrain_test
#SBATCH --output=logs/pretrain_test_%j.out
#SBATCH --error=logs/pretrain_test_%j.err
#SBATCH --partition=REPLACE_ME        # e.g. gpu
#SBATCH --gres=gpu:1                  # or --gres=gpu:0 for a pure CPU smoke test
#SBATCH --cpus-per-task=24            # DataModule uses num_workers=20
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Quick sanity-check run: batch_size=1, no --config, no SLURM array.
# This deliberately does NOT set SLURM_ARRAY_TASK_ID, so pretrainArtificial.py
# falls into its hardcoded default `params` dict (batch_size=1) and prints
# the [CONFIG]/[DATA]/[MODEL]/[TRAIN] sanity messages as it loads/trains.

set -euo pipefail

mkdir -p logs

# --- environment setup ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate REPLACE_ME_CONDA_ENV   # e.g. conda activate criscross

# --- run from repo root so `CRISCross` resolves as a module ---
cd "$(dirname "$0")/.."

srun python -m CRISCross.pretrainArtificial
