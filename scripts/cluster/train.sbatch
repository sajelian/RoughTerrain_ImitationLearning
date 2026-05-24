#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --output=/cluster/scratch/%u/rt_il_logs/slurm-%j.out
#SBATCH --error=/cluster/scratch/%u/rt_il_logs/slurm-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=sajelian@gmx.net
#SBATCH --job-name="rt-il-train-$(date +"%Y-%m-%dT%H:%M")"
#SBATCH --exclude=eu-g6-064
#SBATCH --account=ls_polle

set -e

# Internet access on ETH compute nodes
module load eth_proxy

# === Paths ===
SIF="$HOME/isaac-lab-2.3.2.sif"
SCRATCH="/cluster/scratch/$USER"
RT_IL_DIR="$SCRATCH/RT_IL"
LOG_DIR="$SCRATCH/rt_il_logs"
PIP_CACHE="$SCRATCH/pip_cache"
ISAAC_CACHE="$SCRATCH/docker-isaac-sim"

CONTAINER_TMP="$SCRATCH/container_tmp"

mkdir -p "$LOG_DIR" "$PIP_CACHE" "$CONTAINER_TMP" \
    "$ISAAC_CACHE/cache/kit" "$ISAAC_CACHE/cache/ov" \
    "$ISAAC_CACHE/cache/pip" "$ISAAC_CACHE/cache/glcache" \
    "$ISAAC_CACHE/cache/computecache" "$ISAAC_CACHE/logs" \
    "$ISAAC_CACHE/data" "$ISAAC_CACHE/documents" \
    "$ISAAC_CACHE/kit_data"

# === Run training inside container ===
apptainer exec --nv --writable-tmpfs --cleanenv \
    --bind "$RT_IL_DIR:/workspace/RT_IL" \
    --bind "$LOG_DIR:/workspace/RT_IL/logs" \
    --bind "$CONTAINER_TMP:/tmp:rw" \
    --bind "$ISAAC_CACHE/cache/kit:/isaac-sim/kit/cache:rw" \
    --bind "$ISAAC_CACHE/kit_data:/isaac-sim/kit/data:rw" \
    --bind "$ISAAC_CACHE/cache/ov:/root/.cache/ov:rw" \
    --bind "$ISAAC_CACHE/cache/pip:/root/.cache/pip:rw" \
    --bind "$ISAAC_CACHE/cache/glcache:/root/.cache/nvidia/GLCache:rw" \
    --bind "$ISAAC_CACHE/cache/computecache:/root/.nv/ComputeCache:rw" \
    --bind "$ISAAC_CACHE/logs:/root/.nvidia-omniverse/logs:rw" \
    --bind "$ISAAC_CACHE/data:/root/.local/share/ov/data:rw" \
    --bind "$ISAAC_CACHE/documents:/root/Documents:rw" \
    --env "PIP_CACHE_DIR=/tmp/pip_cache" \
    --env "TMPDIR=/tmp" \
    --env "WANDB_API_KEY=$WANDB_API_KEY" \
    --env "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-}" \
    --env "WARP_CACHE_PATH=/tmp/warp_cache" \
    --env "http_proxy=$http_proxy" \
    --env "https_proxy=$https_proxy" \
    --env "no_proxy=$no_proxy" \
    "$SIF" \
    bash /workspace/RT_IL/scripts/cluster/entrypoint.sh "$@"
