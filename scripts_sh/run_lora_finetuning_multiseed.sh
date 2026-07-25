#!/bin/bash
#SBATCH --job-name=lora_multiseed_loso
#SBATCH --partition=gpu-single
#SBATCH --array=0-16               # 17 guides (indices 0..16)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00            # 10 seeds run sequentially — ~10x longer than single-seed
#SBATCH --output=logs/slurm/lora_multiseed_%A_%a.out
#SBATCH --error=logs/slurm/lora_multiseed_%A_%a.err

set -euo pipefail

# All 17 GuideIDs in the same order they appear when sorted
GUIDES=(sg1 sg10 sg12 sg13 sg16 sg18 sg19 sg2 sg23 sg24 sg26 sg28 sg3 sg5 sg6 sg7 sg8)

TEST_GUIDE=${GUIDES[$SLURM_ARRAY_TASK_ID]}
echo "Running multi-seed LOSO for: $TEST_GUIDE (task $SLURM_ARRAY_TASK_ID)"

# NOTE: logs/slurm/ must exist before sbatch is called — Slurm opens
# the --output/--error files before this script body runs.
# Run once on the cluster: mkdir -p logs/slurm

source /home/fr/fr_fr/fr_js2142/miniforge3/etc/profile.d/conda.sh
conda activate myenv

cd /gpfs/bwfor/work/ws/fr_js2142-minex/pretrain_final/CRISCross || { echo "ERROR: could not cd to CRISCross directory"; exit 1; }

mkdir -p results/lora_multiseed_parts

python examples/lora_finetuning_multiseed.py \
    --test_guide "$TEST_GUIDE" \
    --results_file "results/multiseed_parts/loso_${TEST_GUIDE}.csv" \
    --accelerator gpu


