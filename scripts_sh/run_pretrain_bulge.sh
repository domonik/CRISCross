#!/bin/bash
#SBATCH --job-name=criscross_pretrain_bulge
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/slurm/pretrain_bulge_%A_%a.out
#SBATCH --error=logs/slurm/pretrain_bulge_%A_%a.err

# NOTE: logs/slurm/ must exist before sbatch is called.
# Run once on the cluster: mkdir -p logs/slurm
#
# Bulge-aware pretraining: a single model at bulge_rate=0.2, band_delta=2.
#
# Unlike run_pretrain_full.sh, this passes --config: the hardcoded params dict
# in pretrainArtificial.py has no bulge keys, so running without a config gives
# a mismatch-only run (band_delta=0).
#
#   sbatch scripts_sh/run_pretrain_bulge.sh
#
# To turn this back into the 0/5/20/50/100 % ablation, regenerate the config
# with the default rates and submit it as an array:
#   python examples/make_bulge_sweep_config.py --out configs/bulge_sweep.json
#   sbatch --array=0-4 scripts_sh/run_pretrain_bulge.sh   # with CONFIG_PATH updated

set -euo pipefail

source /home/fr/fr_fr/fr_js2142/miniforge3/etc/profile.d/conda.sh
conda activate myenv

cd /gpfs/bwfor/work/ws/fr_js2142-minex/pretrain_final/CRISCross || { echo "ERROR: could not cd to CRISCross directory"; exit 1; }

CONFIG_PATH="configs/bulge20pct.json"
[[ -f "${CONFIG_PATH}" ]] || python examples/make_bulge_sweep_config.py \
    --out "${CONFIG_PATH}" --rates 0.2 --seeds 0 --experiment PretrainingBulge

# pretrainArtificial.py only reads --config when SLURM_ARRAY_TASK_ID is set, so
# define it for plain (non-array) submissions too.
export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
# Trainer(deterministic=True) needs this for cuBLAS.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

srun python -m CRISCross.pretrainArtificial --config "${CONFIG_PATH}"
