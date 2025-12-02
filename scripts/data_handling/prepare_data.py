"""Script to prepare SMPL-X trajectory data for use in GMR retargeting."""
import numpy as np
from scipy.spatial.transform import Rotation as R

with np.load("trajectories/mesh/smplx_traj.npz", allow_pickle=True) as npz:
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
np.savez("trajectories/mesh/smplx_traj_mod.npz", **data)
