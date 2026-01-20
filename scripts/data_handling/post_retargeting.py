"""Script to post-process retargeted trajectory to align with original trajectory."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-npz",
        type=Path,
        default=Path("trajectories/mesh/motion_prepped.npz"),
        help="Path to the original SMPL-X trajectory npz file.",
    )
    parser.add_argument(
        "--retargeted-csv",
        type=Path,
        default=Path("trajectories/mesh/motion_rt2g1.csv"),
        help="Path to the retargeted motion csv file.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        help="Optional output csv path; defaults to appending _rdy to the retargeted stem.",
    )

    parser.add_argument(
        "--leica-z",
        type=float,
        default=0.0,  # 1.295, 1.3
        help="Optional offset to add to z height to account for lidar height difference.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv or args.retargeted_csv.with_name(args.retargeted_csv.stem + "_rdy.csv")

    with np.load(args.original_npz, allow_pickle=True) as npz:
        data = dict(npz)

    # Load retargeted data
    retargeted_data = pandas.read_csv(args.retargeted_csv, header=None)

    while retargeted_data.shape[0] < data["joints"][:, 0, :].shape[0]:
        retargeted_data = pandas.concat([retargeted_data, retargeted_data.iloc[[-1]]], ignore_index=True)

    overwrite_all = True
    if not overwrite_all:
        # can use this to only offset starting xy
        xy = retargeted_data.iloc[:, :2].to_numpy()
        xy += data["joints"][:, 0, :][0, :2] - xy[0]
        retargeted_data.iloc[:, :2] = xy
    else:
        retargeted_data.iloc[:, :3] = data["joints"][:, 0, :]

    # notice: new FORMAT has z already accounted for but keep this for old data
    # if robot is below the surface in GMR you need the offset!
    retargeted_data.iloc[:, 2] += 0  # args.leica_z

    plot = False
    if plot:
        plt.scatter(data["joints"][:, 0, :][:, 0], data["joints"][:, 0, :][:, 1], s=10, label='original')
        plt.scatter(retargeted_data.iloc[:, 0], retargeted_data.iloc[:, 1], s=1, label='retargeted')
        plt.legend()
        plt.title("Root Position XY")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.axis('equal')
        plt.grid()
        plt.show()

    # save
    retargeted_data.to_csv(output_csv, index=False, header=False)


if __name__ == "__main__":
    main()
