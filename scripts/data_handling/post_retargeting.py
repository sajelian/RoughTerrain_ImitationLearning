"""Script to post-process retargeted trajectory to align with original trajectory."""
import numpy as np
import pandas
import matplotlib.pyplot as plt

with np.load("trajectories/mesh/smplx_traj.npz", allow_pickle=True) as npz:
    data = dict(npz)

# Load retargeted data
retargeted_data = pandas.read_csv('trajectories/mesh/sequence_07.csv', header=None)

while retargeted_data.shape[0] < data["transl"].shape[0]:
    retargeted_data = pandas.concat([retargeted_data, retargeted_data.iloc[[-1]]], ignore_index=True)

overwrite_all = True
if not overwrite_all:
    # can use this to only offset starting xy
    xy = retargeted_data.iloc[:, :2].to_numpy()
    xy += data["transl"][0, :2] - xy[0]
    retargeted_data.iloc[:, :2] = xy
else:
    retargeted_data.iloc[:, :3] = data["transl"][:, :]

retargeted_data.iloc[:, 2] += 1.31  # TODO this is hard coded ass, this must match the config parameter of the .usd -> Alex should fix this on his side tbh

plot = True
if plot:
    plt.scatter(data["transl"][:, 0], data["transl"][:, 1], s=10, label='original')
    plt.scatter(retargeted_data.iloc[:, 0], retargeted_data.iloc[:, 1], s=1, label='retargeted')
    plt.legend()
    plt.title("Root Position XY")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')
    plt.grid()
    plt.show()

# save
retargeted_data.to_csv('trajectories/mesh/sequence_07_rdy.csv', index=False, header=False)
