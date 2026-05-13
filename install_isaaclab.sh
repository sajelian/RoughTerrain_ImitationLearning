#!/usr/bin/env bash
# ==============================================================================
# Reproducible Isaac Lab Installation Script (uv-based)
# ==============================================================================
# Usage:
#   bash install_isaaclab.sh
#   or, one-liner from a hosted URL:
#   curl -fsSL https://your-host/install_isaaclab.sh | bash
#
# Why this script exists:
#   The official docs can change at any time. By pinning every version here
#   (Python, Isaac Sim, PyTorch, Isaac Lab git commit), any team member or
#   CI machine can reproduce an identical environment months later.
#
# Tested on: Ubuntu 22.04 LTS (GLIBC 2.35+), x86_64, CUDA 12.8
# ==============================================================================

set -euo pipefail  # -e: exit on error, -u: error on unset var, -o pipefail: catch pipe errors

# ==============================================================================
# ── PINNED VERSIONS (edit here to upgrade) ────────────────────────────────────
# ==============================================================================

# Python version — Isaac Sim 5.x requires exactly 3.11
PYTHON_VERSION="3.11"

# Isaac Sim pip package version (from pypi.nvidia.com)
ISAAC_SIM_VERSION="5.1.0"

# Isaac Lab: pin to a specific commit SHA for true reproducibility.
# A git tag (e.g. "v2.3.2") can be moved; a SHA never changes.
# Update this SHA when you want to upgrade Isaac Lab.
ISAAC_LAB_REPO="https://github.com/isaac-sim/IsaacLab.git"
ISAAC_LAB_COMMIT="3e73d6dd79080fd7632488c061052a6edd52e230t"  # main as of 2025-05-14

# PyTorch — must match your CUDA version. cu128 = CUDA 12.8
TORCH_VERSION="2.7.0"
TORCHVISION_VERSION="0.22.0"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

# Name of the virtual environment directory
ENV_NAME="env_isaaclab"

# Which Isaac Lab learning frameworks to install.
# Options: all | rl_games | rsl_rl | sb3 | skrl | robomimic | none
ISAACLAB_INSTALL_TARGET="all"

# ==============================================================================
# ── HELPERS ───────────────────────────────────────────────────────────────────
# ==============================================================================

# Print a coloured section header so progress is easy to follow in the terminal
log() {
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════"
}

# Abort with a clear message if a prerequisite is missing
require() {
    command -v "$1" &>/dev/null || {
        echo "ERROR: '$1' not found. $2"
        exit 1
    }
}

# ==============================================================================
# ── PRE-FLIGHT CHECKS ─────────────────────────────────────────────────────────
# ==============================================================================

log "Checking prerequisites"

# uv must be installed — https://docs.astral.sh/uv/getting-started/installation/
require uv "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"

# git is needed to clone Isaac Lab at a specific commit
require git "Install with: sudo apt install git"

# cmake + build-essential are Linux deps for robomimic (bundled in --install all)
if ! dpkg -s cmake build-essential &>/dev/null 2>&1; then
    echo "Installing system deps: cmake, build-essential …"
    sudo apt-get install -y cmake build-essential
fi

# Verify GLIBC >= 2.35 (Isaac Sim hard requirement)
GLIBC_VERSION=$(ldd --version | head -1 | awk '{print $NF}')
GLIBC_MINOR=$(echo "$GLIBC_VERSION" | cut -d. -f2)
if [ "$GLIBC_MINOR" -lt 35 ]; then
    echo "ERROR: GLIBC $GLIBC_VERSION detected. Isaac Sim pip install requires >= 2.35."
    echo "Consider using the Binaries installation path instead."
    exit 1
fi
echo "GLIBC $GLIBC_VERSION ✓"

# ==============================================================================
# ── STEP 1: Create the virtual environment ────────────────────────────────────
# ==============================================================================

log "Creating uv virtual environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"

# --python pins the exact interpreter version
# --seed bootstraps pip into the env — Isaac Lab's install scripts require pip
# even though we use "uv pip" for all our own installs
uv venv --python "${PYTHON_VERSION}" --seed "${ENV_NAME}"

# Activate so all subsequent pip/uv commands target this env
# shellcheck source=/dev/null
source "${ENV_NAME}/bin/activate"

echo "Activated: $(which python) → $(python --version)"

# ==============================================================================
# ── STEP 2: Upgrade pip inside the env ────────────────────────────────────────
# ==============================================================================

log "Upgrading pip"

# uv pip delegates editable installs to pip under the hood, so we keep pip fresh
pip install --upgrade pip

# ==============================================================================
# ── STEP 3: Install Isaac Sim ─────────────────────────────────────────────────
# ==============================================================================

log "Installing Isaac Sim ${ISAAC_SIM_VERSION}"

# [all]        — installs all Isaac Sim modules
# [extscache]  — pre-caches Omniverse extensions so the first launch is faster
# --extra-index-url — NVIDIA hosts Isaac Sim wheels on their own PyPI server
uv pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

# ==============================================================================
# ── STEP 4: Install PyTorch (CUDA-enabled) ────────────────────────────────────
# ==============================================================================

log "Installing PyTorch ${TORCH_VERSION} (CUDA)"

# We install a CUDA-aware build from PyTorch's own index, NOT the default PyPI,
# because the default PyPI torch is CPU-only and will silently break GPU training
uv pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

# ==============================================================================
# ── STEP 5: Clone Isaac Lab at the pinned commit ──────────────────────────────
# ==============================================================================

log "Cloning Isaac Lab @ ${ISAAC_LAB_COMMIT}"

if [ -d "IsaacLab" ]; then
    echo "IsaacLab/ directory already exists — skipping clone."
    echo "If you want a fresh install, delete the directory first."
else
    # Clone without checking out (faster: avoids downloading the full working tree twice)
    git clone "${ISAAC_LAB_REPO}" IsaacLab
fi

cd IsaacLab

# Hard-reset to the pinned commit — this is what makes the install reproducible.
# Unlike a branch pointer, a SHA is immutable.
git fetch origin
git checkout "${ISAAC_LAB_COMMIT}"

echo "Isaac Lab HEAD: $(git log -1 --oneline)"

# ==============================================================================
# ── STEP 6: Install Isaac Lab extensions ─────────────────────────────────────
# ==============================================================================

log "Installing Isaac Lab extensions (target: ${ISAACLAB_INSTALL_TARGET})"

# isaaclab.sh --install iterates over all packages in source/ and installs them
# with "pip install --editable", meaning your source edits are reflected
# immediately without needing to reinstall — great for thesis experiments.
#
# It also detects the active virtual env and uses "uv pip" automatically
# when uv is available.
./isaaclab.sh --install "${ISAACLAB_INSTALL_TARGET}"


# ==============================================================================
# ── STEP 7: Install your project (RT_IL) ─────────────────────────────────────
# ==============================================================================
 
log "Installing RT_IL project"
 
# Install your own project as an editable package (-e) from within the IsaacLab
# source tree. Editable mode means Python imports your actual source files
# directly — no reinstall needed when you change your code during experiments.
# We use "python -m pip" (rather than bare "pip") to guarantee we're using the
# pip that belongs to the currently active virtual environment.
python -m pip install -e ../source/RT_IL

# ==============================================================================
# ── DONE ──────────────────────────────────────────────────────────────────────
# ==============================================================================

log "Installation complete!"

cat <<EOF

  To activate this environment in a new terminal, run:
    source ${ENV_NAME}/bin/activate   # from the directory where you ran this script
    cd IsaacLab

  Verify the install:
    python scripts/tutorials/00_sim/create_empty.py

  Quick RL sanity check (headless):
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
        --task=Isaac-Ant-v0 --headless

  NOTE: If you see "No module named 'isaacsim'", run:
    source _isaac_sim/setup_conda_env.sh
  This sources Isaac Sim's own env hooks (required after every fresh terminal).

EOF