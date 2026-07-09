#!/bin/bash
#SBATCH --job-name=criscross_pretrain_test
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/pretrain_test_%j.out
#SBATCH --error=logs/slurm/pretrain_test_%j.err

# NOTE: logs/slurm/ must exist before sbatch is called.
# Run once on the cluster: mkdir -p logs/slurm

# Quick sanity-check run: batch_size=1, no --config, no SLURM array.
# This deliberately does NOT set SLURM_ARRAY_TASK_ID, so pretrainArtificial.py
# falls into its hardcoded default `params` dict (batch_size=1) and prints
# the [CONFIG]/[DATA]/[MODEL]/[TRAIN] sanity messages as it loads/trains.

set -euo pipefail

source /home/fr/fr_fr/fr_js2142/miniforge3/etc/profile.d/conda.sh
conda activate myenv

cd /gpfs/bwfor/work/ws/fr_js2142-minex/pretrain_final/CRISCross || { echo "ERROR: could not cd to CRISCross directory"; exit 1; }

srun python -m CRISCross.pretrainArtificial