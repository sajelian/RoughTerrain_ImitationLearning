#!/bin/bash

echo "[INFO] Running inside Isaac Lab container"
echo "[INFO] Isaac Sim: $(cat /isaac-sim/VERSION 2>/dev/null || echo 'unknown')"

# Ensure PATH includes isaac-sim python
export PATH="/isaac-sim/kit/python/bin:/isaac-sim:/usr/local/bin:/usr/bin:/bin:$PATH"
export ISAACLAB_PATH=/workspace/isaaclab

# Patch isaaclab_rl 0.54.2 -> 0.54.3: replace rl_cfg.py to add RslRlMLPModelCfg etc.
RSLRL_DIR="/workspace/isaaclab/source/isaaclab_rl/isaaclab_rl/rsl_rl"
if ! grep -q "RslRlMLPModelCfg" "$RSLRL_DIR/rl_cfg.py" 2>/dev/null; then
    cp /workspace/RT_IL/scripts/cluster/compat/rsl_rl/rl_cfg.py "$RSLRL_DIR/rl_cfg.py"
    cp /workspace/RT_IL/scripts/cluster/compat/rsl_rl/utils.py "$RSLRL_DIR/utils.py"
    echo "from .utils import handle_deprecated_rsl_rl_cfg" >> "$RSLRL_DIR/__init__.py"
    echo "[COMPAT] Patched isaaclab_rl.rsl_rl to 0.54.3"
fi

# Fix URDF extension version mismatch (container has 2.4.30, code expects 2.4.31)
# 1. Downgrade version string
# 2. Remove set_merge_fixed_ignore_inertia call (not available in 2.4.30)
sed -i -e 's/urdf-2.4.31/urdf-2.4.30/g' \
       -e '/set_merge_fixed_ignore_inertia/d' \
    /workspace/isaaclab/source/isaaclab/isaaclab/sim/converters/urdf_converter.py 2>/dev/null || true

# Upgrade rsl-rl to match local version (container has <4.0.0, code requires >=4.0.0)
/isaac-sim/python.sh -m pip install "rsl-rl-lib==5.0.1" 2>&1

# Install RT_IL (IsaacLab extensions are pre-installed in the image)
/isaac-sim/python.sh -m pip install --editable /workspace/RT_IL/source/RT_IL 2>&1

echo "[INFO] Starting training..."
cd /workspace/RT_IL
/isaac-sim/python.sh "$@" 2>&1
EXIT_CODE=$?
echo "[INFO] Training exited with code: $EXIT_CODE"
exit $EXIT_CODE
