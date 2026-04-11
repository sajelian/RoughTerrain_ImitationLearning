"""Offset the root position of a retargeted motion CSV by a fixed (x, y, z) vector.

Usage:
    python scripts/data_handling/offset_root.py --csv motion.csv --x 1.0 --y 0.5 --z 0.1
    python scripts/data_handling/offset_root.py --csv motion.csv --x 1.0 --y 0.5 --z 0.1 --output motion_shifted.csv
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Add a fixed offset to root x/y/z in a retargeted motion CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to the retargeted motion CSV.")
    parser.add_argument("--x", type=float, default=0.0, help="Offset to add to root x.")
    parser.add_argument("--y", type=float, default=0.0, help="Offset to add to root y.")
    parser.add_argument("--z", type=float, default=0.0, help="Offset to add to root z.")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path (defaults to <stem>_shifted.csv).")
    args = parser.parse_args()

    data = np.loadtxt(args.csv, delimiter=",")
    print(f"Loaded {data.shape[0]} frames from {args.csv}")

    data[:, 0] += args.x
    data[:, 1] += args.y
    data[:, 2] += args.z

    output = args.output or args.csv.with_name(args.csv.stem + "_shifted.csv")
    np.savetxt(output, data, delimiter=",")
    print(f"Saved to {output} (offset: x={args.x}, y={args.y}, z={args.z})")


if __name__ == "__main__":
    main()
