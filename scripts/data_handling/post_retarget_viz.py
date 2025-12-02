import numpy as np
import pandas
import matplotlib.pyplot as plt
import pathlib
import wandb

# with np.load("artifacts/trans_fixed_offset:v1/motion.npz", allow_pickle=True) as npz:
#     data = dict(npz)


api = wandb.Api()
artifact = api.artifact("sajelian-eth-zurich-org/wandb-registry-Motions/inv1:latest")
data = np.load(pathlib.Path(artifact.download()) / "motion.npz")

retargeted_og = pandas.read_csv('trajectories/mesh/sequence_07.csv')
retargeted_mod = pandas.read_csv('trajectories/mesh/sequence_07_rdy.csv')

print(data.keys())
print(data["body_lin_vel_w"].shape)
print(retargeted_og.iloc[:, 0:3].shape)
print(retargeted_mod.iloc[:-2, 0:3].shape)


plt.plot(retargeted_og.iloc[0:, 0:3])
plt.plot(retargeted_mod.iloc[0:-2, 0:3], label=['x', 'y', 'z'])
plt.plot(data["body_pos_w"][:, 0, :], label=['x_npz', 'y_npz', 'z_npz'])  # (frame, body_index, xyz vel)
plt.legend()
plt.show()
