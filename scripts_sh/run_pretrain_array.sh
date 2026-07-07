#!/bin/bash
set -euo pipefail
#SBATCH --job-name=criscross_pretrain
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=48:00:00                # adjust to how long a full pretraining run needs
#SBATCH --array=0-REPLACE_ME           # one task per entry in the config JSON (see note below)
#SBATCH --output=logs/slurm/pretrain_%A_%a.out
#SBATCH --error=logs/slurm/pretrain_%A_%a.err

# NOTE: logs/slurm/ must exist before sbatch is called.
# Run once on the cluster: mkdir -p logs/slurm
#
# To find the array upper bound, count entries in the config file, e.g.:
#   python -c "import json; print(len(json.load(open('configs/artificial_AG_param_combinationsWTC.json'))) - 1)"

# Full production run: reads configs/artificial_AG_param_combinationsWTC.json and
# runs the config[SLURM_ARRAY_TASK_ID] entry. Because SLURM_ARRAY_TASK_ID is set
# here (via --array), pretrainArtificial.py takes the --config branch instead of
# the hardcoded test defaults.

source /home/fr/fr_fr/fr_js2142/miniforge3/etc/profile.d/conda.sh
conda activate myenv

cd /gpfs/bwfor/work/ws/fr_js2142-minex/CRISCross || { echo "ERROR: could not cd to CRISCross directory"; exit 1; }

CONFIG_PATH="configs/artificial_AG_param_combinationsWTC.json"

srun python -m CRISCross.pretrainArtificial --config "${CONFIG_PATH}"
