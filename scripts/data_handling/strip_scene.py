"""Strip scene mesh, images, and point cloud from an npz file, keeping only trajectory data."""
import argparse
from pathlib import Path

import numpy as np

# Keys to remove (scene mesh, RGB images, point cloud)
KEYS_TO_STRIP = {
    "scene_vertices",
    "scene_faces",
    "scene_normals",
    "scene_vertex_colors",
    "aria_rgb_imgs",
    "aria_obs_rgb_imgs",
    "aria_pcd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the source npz file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path; defaults to <input_stem>_stripped.npz in the same directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input.with_name(args.input.stem + "_stripped.npz")

    with np.load(args.input, allow_pickle=True) as npz:
        kept = {k: npz[k] for k in npz.files if k not in KEYS_TO_STRIP}
        stripped = [k for k in npz.files if k in KEYS_TO_STRIP]

    print(f"Stripped {len(stripped)} keys: {stripped}")
    print(f"Keeping {len(kept)} keys: {list(kept.keys())}")

    np.savez(output_path, **kept)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
