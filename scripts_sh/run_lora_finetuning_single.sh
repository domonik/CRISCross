#!/bin/bash
#SBATCH --job-name=lora_single_sanity
#SBATCH --partition=devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00            # single guide, single seed - quick sanity check
#SBATCH --output=logs/slurm/lora_single_%j.out
#SBATCH --error=logs/slurm/lora_single_%j.err

set -euo pipefail

# Sanity-check run: ONE guide, ONE seed. Run this before launching the full
# multi-guide/multi-seed sweep (run_lora_finetuning_multiseed.sh) to catch
# config/data/checkpoint problems quickly and cheaply.
#
# Usage:
#   sbatch run_lora_finetuning_single.sh [TEST_GUIDE] [SEED]
#   (defaults: TEST_GUIDE=sg1, SEED=0)

TEST_GUIDE=${1:-sg1}
SEED=${2:-0}

echo "Running LoRA-finetuning sanity check for: $TEST_GUIDE (seed=$SEED)"

# NOTE: logs/slurm/ must exist before sbatch is called — Slurm opens
# the --output/--error files before this script body runs.
# Run once on the cluster: mkdir -p logs/slurm

source /home/fr/fr_fr/fr_js2142/miniforge3/etc/profile.d/conda.sh
conda activate myenv

cd /gpfs/bwfor/work/ws/fr_js2142-minex/pretrain_final/CRISCross || { echo "ERROR: could not cd to CRISCross directory"; exit 1; }

mkdir -p results

python examples/lora_finetuning_single.py \
    --test_guide "$TEST_GUIDE" \
    --seed "$SEED" \
    --results_file "results/lora_sanity_check.csv" \
    --accelerator gpu

