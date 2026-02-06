"""Script to prepare SMPL-X trajectory data for use in GMR retargeting. Note: this is subject to change based on input data format."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-file",
        type=Path,
        default=Path("trajectories/mesh/motion.npz"),
        help="Path to the source SMPL-X trajectory npz file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional output path; defaults to appending _prepped to the input stem.",
    )
    parser.add_argument(
        "--frame-range",
        type=str,
        default=None,
        help="Optional frame range as START-END (e.g. 0-100 or 100-550). Both inclusive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output_path or args.motion_file.with_name(args.motion_file.stem + "_prepped.npz")

    with np.load(args.motion_file, allow_pickle=True) as npz:
        data = dict(npz)

    for key in ("betas", "global_orient", "body_pose", "transl", "joints"):
        if key not in data:
            smplx_key = f"smplx_{key}"
            smplx_egomc_key = f"smplx_egomc_{key}"
            if smplx_key in data:
                data[key] = data.pop(smplx_key)
            elif smplx_egomc_key in data:
                data[key] = data.pop(smplx_egomc_key)
            else:
                raise KeyError(f"Key {key} not found in the input npz file.")

    # optionally slice to requested frame range
    if args.frame_range is not None:
        start, end = args.frame_range.split("-")
        start, end = int(start), int(end) + 1  # inclusive end
        for key in ("global_orient", "body_pose", "transl", "joints"):
            if key in data and data[key].ndim >= 1:
                data[key] = data[key][start:end]

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
    data["gender"] = "male"  # TODO check if final data has this, then drop it

    # save
    np.savez(output_path, **data)


if __name__ == "__main__":
    main()
