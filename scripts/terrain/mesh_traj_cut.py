import numpy as np
import pandas
import trimesh

trajectory = pandas.read_csv('trajectories/mesh/sequence_07_rdy.csv')
mesh = trimesh.load_mesh('source/RT_IL/RT_IL/assets/meshes/sequence_07.obj')
mesh.invert()

# Offset the mesh by translation, TODO this should not be needed if Alex applies the translation on his end
apply_translation = True
if apply_translation:
    mesh.apply_translation((0, 0, 1.47))

trajectory_path = trimesh.load_path(trajectory.iloc[:, 0:3].to_numpy())
# color the trajectory path
trajectory_path.colors = [[255, 247, 0, 255]]
# mark the start and end points with green and red spheres # TODO could also do cones and use orientation aswell
start_sphere = trimesh.creation.icosphere(radius=0.1, subdivisions=3)
start_sphere.apply_translation(trajectory.iloc[0, 0:3].to_numpy())
start_sphere.visual.face_colors = [0, 255, 0, 255]
end_sphere = trimesh.creation.icosphere(radius=0.1, subdivisions=3)
end_sphere.apply_translation(trajectory.iloc[-1, 0:3].to_numpy())
end_sphere.visual.face_colors = [255, 0, 0, 255]

tube_radius = 1.32
traj_points = trajectory.iloc[:, 0:3].to_numpy()
segment_starts = traj_points[:-1]
segment_ends = traj_points[1:]
segment_dirs = segment_ends - segment_starts
segment_length_sq = np.einsum('ij,ij->i', segment_dirs, segment_dirs)
safe_dirs = segment_dirs.copy()
safe_dirs[segment_length_sq == 0.0] = 0.0
segment_length_sq = np.where(segment_length_sq == 0.0, 1.0, segment_length_sq)

vertices = mesh.vertices
inside_vertices = np.zeros(len(vertices), dtype=bool)
chunk_size = 2048
radius_sq = tube_radius * tube_radius

for start in range(0, len(vertices), chunk_size):
    verts_chunk = vertices[start:start + chunk_size]
    rel = verts_chunk[:, None, :] - segment_starts[None, :, :]
    proj = np.einsum('ijk,jk->ij', rel, safe_dirs) / segment_length_sq
    proj = np.clip(proj, 0.0, 1.0)
    closest = segment_starts + proj[:, :, None] * safe_dirs
    deltas = verts_chunk[:, None, :] - closest
    min_dist_sq = np.einsum('ijk,ijk->ij', deltas, deltas).min(axis=1)
    inside_vertices[start:start + chunk_size] = min_dist_sq <= radius_sq

faces_mask = inside_vertices[mesh.faces].all(axis=1)
mesh = mesh.submesh([faces_mask], append=True, repair=False) if faces_mask.any() else trimesh.Trimesh()

if len(mesh.vertices):
    vertices = mesh.vertices
    below_traj = np.zeros(len(vertices), dtype=bool)

    for start in range(0, len(vertices), chunk_size):
        verts_chunk = vertices[start:start + chunk_size]
        rel = verts_chunk[:, None, :] - segment_starts[None, :, :]
        proj = np.einsum('ijk,jk->ij', rel, safe_dirs) / segment_length_sq
        proj = np.clip(proj, 0.0, 1.0)
        closest = segment_starts + proj[:, :, None] * safe_dirs
        deltas = verts_chunk[:, None, :] - closest
        dist_sq = np.einsum('ijk,ijk->ij', deltas, deltas)
        min_idx = dist_sq.argmin(axis=1)
        closest_z = closest[np.arange(len(verts_chunk)), min_idx, 2]
        below_traj[start:start + chunk_size] = verts_chunk[:, 2] <= closest_z

    faces_mask = below_traj[mesh.faces].all(axis=1)
    mesh = mesh.submesh([faces_mask], append=True, repair=False) if faces_mask.any() else trimesh.Trimesh()

scene = trimesh.Scene([mesh, trajectory_path, start_sphere, end_sphere])
scene.show()  # type: ignore
mesh.export('source/RT_IL/RT_IL/assets/meshes/sequence_07_fix.obj')  # type: ignore
