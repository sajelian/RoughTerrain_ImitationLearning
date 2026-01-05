"""Script to prepare SMPL-X trajectory data for use in GMR retargeting. Note: this is subject to change based on input data format."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion_file",
        type=Path,
        default=Path("trajectories/mesh/motion.npz"),
        help="Path to the source SMPL-X trajectory npz file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional output path; defaults to appending _prepped to the input stem.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path or args.motion_file.with_name(args.motion_file.stem + "_prepped.npz")

    with np.load(args.motion_file, allow_pickle=True) as npz:
        data = dict(npz)

    # squash betas, they are all the same across frames and pad to be size (16,)
    data["betas"] = data["betas"][0]
    data["betas"] = np.pad(data["betas"], (0, 6))

    # rot_mat -> axis-angle for root_orient
    data["global_orient"] = R.from_matrix(data["global_orient"]).as_rotvec()
    # rot_mat -> axis-angle for pose_body
    body_pose = data["body_pose"]
    data["body_pose"] = R.from_matrix(body_pose.reshape(-1, 3, 3)).as_rotvec().reshape(body_pose.shape[0], -1)

    # add missing frame_rate key
    data["mocap_frame_rate"] = 30.0

    # rename and add keys to match gmr expected input
    data["pose_body"] = data.pop("body_pose")
    data["root_orient"] = data.pop("global_orient")
    data["trans"] = data.pop("transl")
    data["gender"] = "male"

    # save
    np.savez(output_path, **data)


if __name__ == "__main__":
    main()
