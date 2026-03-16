from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils import configclass


def flat_with_helper_box(
    difficulty: float,
    cfg: FlatWithHelperBoxTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Flat terrain tile with an optional scaffold box.

    The box height interpolates linearly from ``initial_box_height`` (at difficulty=0,
    the easiest level) down to zero (at difficulty=1, the hardest level).
    ``box_x`` / ``box_y`` offset the box centre from the tile's env-origin
    (which sits at the tile centre).

    The terrain tile extends from (0, 0, 0) to (size[0], size[1], 0) in local
    coordinates.  The env-origin (robot spawn) is placed at (size[0]/2, size[1]/2, 0).
    """
    box_height = cfg.initial_box_height * max(0.0, 1.0 - difficulty)

    # Tile centre == env-origin; robots spawn here
    cx, cy = cfg.size[0] / 2.0, cfg.size[1] / 2.0
    origin = np.array([cx, cy, 0.0])

    meshes: list[trimesh.Trimesh] = []

    # Flat ground slab (thin; top face at z=0)
    ground = trimesh.creation.box(extents=[cfg.size[0], cfg.size[1], 0.1])
    ground.apply_translation([cx, cy, -0.05])
    meshes.append(ground)

    # Helper box — only add when height is non-trivial
    if box_height > 1e-3:
        box = trimesh.creation.box(extents=[cfg.box_length, cfg.box_width, box_height])
        # box_x / box_y are offsets from the tile's env-origin
        box.apply_translation([cx + cfg.box_x, cy + cfg.box_y, box_height / 2.0])
        meshes.append(box)

    return meshes, origin


@configclass
class FlatWithHelperBoxTerrainCfg(SubTerrainBaseCfg):
    """Flat ground tile with a scaffold box whose height shrinks with difficulty.

    Set as a sub-terrain inside a ``TerrainGeneratorCfg`` with ``num_rows`` equal
    to the desired number of curriculum levels.  At difficulty=0 (row 0) the box
    has full height; at difficulty=1 (last row) the box is gone.
    """

    function = flat_with_helper_box

    initial_box_height: float = MISSING
    """Full height of the scaffold box (metres).  Should match stepping-stone height."""

    box_length: float = MISSING
    """Box dimension along X (metres)."""

    box_width: float = MISSING
    """Box dimension along Y (metres)."""

    box_x: float = 0.0
    """X offset of box centre from the tile's env-origin (metres)."""

    box_y: float = 0.0
    """Y offset of box centre from the tile's env-origin (metres)."""
