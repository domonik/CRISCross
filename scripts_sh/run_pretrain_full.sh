#!/bin/bash
#SBATCH --job-name=criscross_pretrain_full
#SBATCH --partition=gpu-single
#SBATCH --gres=gpu:h200:2             # verify exact GRES string with: sinfo -p gpu-single -o "%N %G" | grep -i h200
#SBATCH --cpus-per-task=48            # ~24 per GPU-rank: 2 DDP ranks x num_workers=20 each = 40 worker procs
#SBATCH --mem=128G
#SBATCH --time=48:00:00               # adjust based on how long the smoke test's per-step timing suggests
#SBATCH --output=logs/slurm/pretrain_full_%j.out
#SBATCH --error=logs/slurm/pretrain_full_%j.err

# NOTE: logs/slurm/ must exist before sbatch is called.
# Run once on the cluster: mkdir -p logs/slurm
#
# 2x H200 GPUs on a single node. No --ntasks-per-node needed -- Lightning's own
# DDP subprocess launcher spawns the 2nd rank itself (torch.cuda.device_count()
# auto-detects both GPUs); srun still only runs one task here.
# batch_size=1024 (per-GPU) x 2 GPUs = 2048 global batch per accumulation unit;
# accumulate_grad_batches=12 in pretrainArtificial.py's params dict keeps the
# effective batch/optimizer-step close to the original 1-GPU value (~25,600).


set -euo pipefail

source /home/fr/fr_fr/fr_js2142/miniforge3/etc/profile.d/conda.sh
conda activate myenv

cd /gpfs/bwfor/work/ws/fr_js2142-minex/pretrain_cris/CRISCross || { echo "ERROR: could not cd to CRISCross directory"; exit 1; }

srun python -m CRISCross.pretrainArtificial
