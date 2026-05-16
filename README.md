<p align="center">
  <img src="docs/images/demo.gif" alt="G1 rough terrain motion tracking demo"/>
</p>

# Rough Terrain Traversal Through Imitation Learning Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

<!-- [[Website]](https://beyondmimic.github.io/) -->
<!-- [[Arxiv]](https://arxiv.org/abs/2508.08241) -->
<!-- [[Video]](https://youtu.be/RS_MtKVIAzY) -->

## Overview

This repository implements motion tracking over rough terrains for a Unitree G1 robot using feature-based imitation learning. (It acts as the expert-policy collection subcomponent of the HumanoidRTT framework. TBD)

## Installation

### Prerequisites

- Ubuntu 22.04 (GLIBC ≥ 2.35), x86_64, CUDA 13.0
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Clone the repository

```bash
git clone git@github.com:sajelian/RT_IL.git
cd RT_IL
```

### Pull robot description assets from GCS

```bash
curl -L -o unitree_description.tar.gz \
  https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/RT_IL/RT_IL/assets/ && \
rm unitree_description.tar.gz
```

### Install Isaac Lab + RT_IL

Run the installation script from the repo root. It creates a `uv` virtual environment (`env_isaaclab/`), installs Isaac Sim, PyTorch, Isaac Lab (pinned commit), and the RT_IL package in editable mode:

```bash
bash install_isaaclab.sh
```

Activate the environment in any new terminal before running scripts:

```bash
source env_isaaclab/bin/activate
```

## Data Preparation

### WandB setup

Log in to your WandB account and create a Registry collection named **Motions** (artifact type: *All Types*) under your organisation. Export your organisation name:

```bash
export WANDB_ENTITY=<your-org-name>
```

### Motion + terrain preprocessing pipeline

`scripts/prep_data.sh` runs the full 5-step pipeline: SMPL-X preparation → GMR retargeting → post-processing → mesh cutting → USD conversion.

```bash
bash scripts/prep_data.sh \
  --motion-npz data/motions/raw/<motion>.npz \
  --mesh data/terrains/raw/<terrain>.obj
```

Run `bash scripts/prep_data.sh --help` for all options (frame ranges, output directories, skipping mesh, etc.).

**Prerequisites for the pipeline:**
- The [GMR](https://github.com/sajelian/GMR) repo must be cloned and its conda environment set up (follow the GMR README). The env is expected at `~/GMR` (or set `GMR_REPO=<path>`). The conda environment name defaults to `gmr_custom` — if yours differs, edit the `ENV_GMR` variable at the top of `scripts/prep_data.sh`
- `IsaacLab/scripts/tools/convert_mesh.py` must be reachable at the path set in the script header.

## Training Expert Policies

```bash
python scripts/rsl_rl/train_mimic_mesh.py \
  --registry_name <your-org>-org/wandb-registry-Motions/<motion_name> \
  --usd_name <usd_mesh_name>
```
## Policy Evaluation

```bash
python scripts/rsl_rl/play_mimic_mesh_ref_viz.py \
  --registry_name <your-org>-org/wandb-registry-Motions/<motion_name> \
  --usd_name <usd_mesh_name> \
  --num_envs 1
```

## Rollout Collection (for Distillation)

```bash
python scripts/rsl_rl/collect_rollouts.py \
  --registry_name <your-org>-org/wandb-registry-Motions/<motion_name> \
  --usd_name <usd_mesh_name>
```
