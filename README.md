# Rough Terrain Traversal Through Imitation Learning Code
[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.1-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

<!-- [[Website]](https://beyondmimic.github.io/) -->
<!-- [[Arxiv]](https://arxiv.org/abs/2508.08241) -->
<!-- [[Video]](https://youtu.be/RS_MtKVIAzY) -->

## Overview
This repository implements the mimicing of motions over rough terrains for a Unitree G1 robot using feature-based imitation learning. It acts as a expert-policy collecting subcomponent of the HumanoidRTT framework.

<!-- BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers. -->

<!-- This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.

For sim-to-sim and sim-to-real deployment, please refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller). -->


## Installation

- Install Isaac Lab v2.3.1 (version must support MultiMeshRayCaster) by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:sajelian/RT_IL.git

# Option 2: HTTPS
git clone https://github.com/sajelian/RT_IL.git
```

- Pull the robot description files from GCS

```bash
# Enter the repository
cd RT_IL
# Rename all occurrences of RT_IL (in files/directories) to your_fancy_extension_name
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/RT_IL/RT_IL/assets/ && \
rm unitree_description.tar.gz
```

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/RT_IL
```

## Training Expert Policies

### Motion and Terrain Preprocessing

First log in to your WandB account; access Registry under Core on the left. Create a new registry collection with the name "
  Motions" and artifact type "All Types".

To prepare the motion and terrain data for training, run the following script:

```bash bash ./scripts/prep_data.sh --motion-npz data/motions/raw/<motion>.npz --leica-z <z-offset> --headless --mesh data/terrains/raw/<terrain>.obj
./scripts/prep_data.sh --motion-npz data/motions/raw/<motion>.npz --leica-z <z-offset> --headless --mesh data/terrains/raw/<terrain>.obj
```

- Debugging
    - Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    - If /tmp folder is not accessible, modify csv_to_npz.py L319 & L326 to a temporary folder of your choice.

### Policy Training

- Train policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--headless --logger wandb --log_project_name {project_name} --run_name {run_name}
```

``` bash
python scripts/rsl_rl/train_mimic_mesh.py --registry_name={your-organization}-org/wandb-registry-Motions/{motion_name} --usd_name {usd_mesh_file} --num_envs {e.g. 4096} --logger wandb --log_project_name {project_name}
```

### Policy Evaluation

- Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play_mimic_mesh_ref_viz.py --registry_name={your-organization}-org/wandb-registry-Motions/{motion_name} --usd_name {usd_mesh_file} --num_envs {e.g. 1}
```
