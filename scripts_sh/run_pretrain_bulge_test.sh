#!/bin/bash
#SBATCH --job-name=criscross_pretrain_bulge_test
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/pretrain_bulge_test_%j.out
#SBATCH --error=logs/slurm/pretrain_bulge_test_%j.err

# NOTE: logs/slurm/ must exist before sbatch is called.
# Run once on the cluster: mkdir -p logs/slurm
#
# Smoke test for the bulge-aware pretraining path. Same resources as
# run_pretrain_test.sh (1 GPU, 24 CPUs, 32G, 2h), but it goes through
# configs/bulge_smoke.json instead of the hardcoded params dict, so the
# bulge sampler is actually exercised: batch_size=1, bulge_rate=0.2,
# band_delta=2, 20 optimizer steps.
#
#   sbatch scripts_sh/run_pretrain_bulge_test.sh
#
# Watch for, in order: the [DATA] bulge-aware sampler line naming
# BulgeGenomicDataset, [DATA] setup() complete, [MODEL] parameter count, and
# [TRAIN] first batch ran successfully with a finite loss. Reaching that last
# line means collate, masking, the alignment gather and the backward pass all
# work end to end. The checkpoint monitor is train_loss, which is epoch-level,
# so a 20-step run logs no monitored value -- that warning is expected here.

set -euo pipefail

source /home/fr/fr_fr/fr_js2142/miniforge3/etc/profile.d/conda.sh
conda activate myenv

cd /gpfs/bwfor/work/ws/fr_js2142-minex/pretrain_final/CRISCross || { echo "ERROR: could not cd to CRISCross directory"; exit 1; }

CONFIG_PATH="configs/bulge_smoke.json"

# pretrainArtificial.py only reads --config when SLURM_ARRAY_TASK_ID is set.
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
# Trainer(deterministic=True) needs this for cuBLAS.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

srun python -m CRISCross.pretrainArtificial --config "${CONFIG_PATH}"
