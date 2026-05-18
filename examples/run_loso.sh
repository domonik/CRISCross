#!/bin/bash
#SBATCH --job-name=lora_loso
#SBATCH --partition=gpu-single
#SBATCH --array=0-16               # 17 guides (indices 0..16)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/loso_%A_%a.out
#SBATCH --error=logs/slurm/loso_%A_%a.err

# All 17 GuideIDs in the same order they appear when sorted
GUIDES=(sg1 sg10 sg12 sg13 sg16 sg18 sg19 sg2 sg23 sg24 sg26 sg28 sg3 sg5 sg6 sg7 sg8)

TEST_GUIDE=${GUIDES[$SLURM_ARRAY_TASK_ID]}
echo "Running leave-one-guide-out for: $TEST_GUIDE (task $SLURM_ARRAY_TASK_ID)"

mkdir -p logs/slurm

source /home/fr/fr_fr/fr_js2142/miniforge3/etc/profile.d/conda.sh
conda activate myenv   

cd /gpfs/bwfor/work/ws/fr_js2142-minex/CRISCross     
python examples/lora_finetuning_all.py \
    --test_guide "$TEST_GUIDE" \
    --results_file results/loso_results.csv \
    --accelerator gpu
