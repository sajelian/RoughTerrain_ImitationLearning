"""Extract the terrain mesh embedded in a captured motion sequence into an OBJ file.

The raw motion ``.npz`` stores the scanned scene geometry alongside the SMPL-X motion in
the ``scene_vertices`` / ``scene_faces`` / ``scene_normals`` / ``scene_vertex_colors``
arrays. This script writes that geometry out as a standalone OBJ.
"""
import argparse
from pathlib import Path

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-npz", type=Path, required=True, help="Raw motion .npz containing the scene_* arrays.")
    parser.add_argument("--output-obj", type=Path, required=True, help="Destination OBJ path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.motion_npz, allow_pickle=True)

    mesh = trimesh.Trimesh(
        vertices=data["scene_vertices"],
        faces=data["scene_faces"],
        vertex_normals=data["scene_normals"],
        vertex_colors=data["scene_vertex_colors"],
        process=False,
    )
    mesh.invert()

    args.output_obj.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output_obj)
    print(f"[INFO]: Terrain mesh written to {args.output_obj}")


if __name__ == "__main__":
    main()
