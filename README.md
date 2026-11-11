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

This repository implements motion tracking over rough terrains for a Unitree G1 robot
using feature-based imitation learning.

The full workflow is a four-stage pipeline. Each stage consumes the output of the
previous one:

| Stage | Script | In → Out |
| --- | --- | --- |
| 1. Data preparation | `scripts/prep_data.sh` | raw motion `.npz` → retargeted trajectory `.csv` + terrain `.usd` |
| 2. Motion packaging | `scripts/csv_to_npz_terrain.py` | trajectory `.csv` (+ USD) → motion `.npz`, logged to a W&B registry |
| 3. Training | `scripts/rsl_rl/train_mimic_mesh.py` | motion `.npz` (from W&B) → trained expert policy |
| 4. Evaluation | `scripts/rsl_rl/play_mimic_mesh_ref_viz.py` | policy checkpoint → reference-overlay visualization |

## Installation

### Prerequisites

- Ubuntu 22.04 (GLIBC ≥ 2.35), x86_64, CUDA 13.0
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Clone the repository

```bash
git clone git@github.com:leggedrobotics/terrain_imitation.git
cd terrain_imitation
```

### Pull robot description assets from GCS

```bash
curl -L -o unitree_description.tar.gz \
  https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/RT_IL/RT_IL/assets/ && \
rm unitree_description.tar.gz
```

> **Note:** `urdf/g1/main.urdf` and `urdf/g1/viz_robot.urdf` are tracked in this repo and will overwrite the downloaded versions. Run `git checkout -- source/RT_IL/RT_IL/assets/unitree_description/urdf/g1/` after extracting the tarball if git doesn't restore them automatically.

### Install Isaac Lab + RT_IL

Run the installation script from the repo root. It creates a `uv` virtual environment (`env_isaaclab/`), installs Isaac Sim, PyTorch, Isaac Lab (pinned commit), and the RT_IL package in editable mode:

```bash
bash install_isaaclab.sh
```

Activate the environment in any new terminal before running scripts:

```bash
source env_isaaclab/bin/activate
```

## Weights & Biases setup

Stages 2–4 use a [Weights & Biases](https://wandb.ai/) registry to store and retrieve the
packaged motions. Log in (`wandb login`) and create a Registry collection named
**Motions** (artifact type: *All Types*) under your entity. Wherever the docs use
`<wandb-entity>`, substitute your own W&B entity (org or username).

## Stage 1 — Data preparation

`scripts/prep_data.sh` turns a raw captured sequence into a retargeted trajectory CSV
and a collision-ready terrain USD. Internally it runs: SMPL-X preparation → GMR
retargeting → post-processing → terrain mesh extraction → mesh cutting → USD conversion.

```bash
bash scripts/prep_data.sh --motion-npz data/motions/raw/<sequence>.npz
```

The raw sequence `.npz` already embeds the scanned terrain (`scene_vertices`/`scene_faces`/…),
so the mesh is extracted automatically. Pass `--mesh <terrain>.obj` to override it with an
external mesh, or `--skip-mesh` to produce the trajectory only. Run
`bash scripts/prep_data.sh --help` for all options (frame ranges, output dirs, etc.).

**Prerequisites:**
- The [GMR](https://github.com/sajelian/GMR) repo cloned with its conda environment set up
  (follow the GMR README). Expected at `~/GMR` (or set `GMR_REPO=<path>`). The conda env
  defaults to `gmr_custom` — if yours differs, edit `ENV_GMR` at the top of `scripts/prep_data.sh`.

## Stage 2 — Motion packaging

Convert the prepared trajectory CSV (and its terrain USD) into a motion `.npz` and log it
to the W&B **Motions** registry:

```bash
python scripts/csv_to_npz_terrain.py \
  --input_file data/motions/prepared/<sequence>_rt2g1_rdy.csv \
  --output_name <motion_name> \
  --usd_name <usd_mesh_name> \
  --headless
```

## Stage 3 — Training expert policies

```bash
python scripts/rsl_rl/train_mimic_mesh.py \
  --registry_name <wandb-entity>/wandb-registry-Motions/<motion_name> \
  --usd_name <usd_mesh_name> \
  --num_envs 2048 --headless --logger wandb --log_project_name RT_IL
```

Use task `Tracking-Mesh-G1-v0` (default) for clean-sim training, or
`Tracking-Mesh-Deploy-G1-v0` (`--task`) for the harder, sim-to-real-oriented variant with
added perturbations and a noised height-scan.

## Stage 4 — Policy evaluation

```bash
python scripts/rsl_rl/play_mimic_mesh_ref_viz.py \
  --registry_name <wandb-entity>/wandb-registry-Motions/<motion_name> \
  --usd_name <usd_mesh_name> \
  --num_envs 1
```

## Acknowledgements

This project was originally built on top of [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking),
which is licensed under the MIT License. The codebase has since been heavily modified.
See `NOTICE.md` for the original license notice.
