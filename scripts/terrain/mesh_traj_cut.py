import argparse
import os

import numpy as np
import pandas
import trimesh


def strip_obj_materials(obj_path):
    mtl_files = []
    try:
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as obj_file:
            lines = obj_file.readlines()
    except OSError:
        return mtl_files

    filtered = []
    for line in lines:
        if line.startswith("mtllib "):
            mtl_name = line.split(None, 1)[1].strip()
            if mtl_name:
                mtl_files.append(mtl_name)
            continue
        if line.startswith("usemtl "):
            continue
        filtered.append(line)

    if filtered != lines:
        with open(obj_path, "w", encoding="utf-8") as obj_file:
            obj_file.writelines(filtered)

    return mtl_files


def cleanup_material_sidecars(obj_path, mtl_files):
    output_dir = os.path.dirname(obj_path)
    for mtl_name in mtl_files:
        if os.path.basename(mtl_name) != mtl_name:
            continue
        mtl_path = os.path.join(output_dir, mtl_name)
        if not os.path.isfile(mtl_path):
            continue

        textures = []
        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as mtl_file:
                for line in mtl_file:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("map_"):
                        parts = line.split()
                        if len(parts) >= 2:
                            textures.append(parts[-1])
        except OSError:
            textures = []

        os.remove(mtl_path)
        for tex_name in textures:
            if os.path.basename(tex_name) != tex_name:
                continue
            tex_path = os.path.join(output_dir, tex_name)
            if os.path.isfile(tex_path):
                os.remove(tex_path)


parser = argparse.ArgumentParser(description="Cut mesh geometry around a trajectory tube.")
parser.add_argument("--trajectory", required=True, help="CSV file containing the trajectory.")
parser.add_argument("--mesh", required=True, help="Mesh file to crop.")
parser.add_argument(
    "--output",
    default=None,
    help="Output mesh path. Defaults to <mesh_basename>_cropped.obj in the same directory as the input mesh.",
)

parser.add_argument(
    "--leica-z",
    type=float,
    default=0.0,  # 1.295, 1.3
    help="Optional offset to add to z height to account for lidar height difference.",
)

parser.add_argument(
    "--headless",
    default=False,
    action="store_true",
    help="Render the mesh scene for visualization.",
)

args = parser.parse_args()

if args.output:
    output_path = args.output
else:
    trajectory_name = os.path.splitext(os.path.basename(args.trajectory))[0]
    mesh_name = os.path.splitext(os.path.basename(args.mesh))[0]
    output_path = os.path.join(os.path.dirname(args.mesh), f"{trajectory_name}_{mesh_name}.obj")

trajectory = pandas.read_csv(args.trajectory)
mesh = trimesh.load_mesh(args.mesh)
show_scene = True

# Offset the mesh by translation, TODO this should not be needed if Alex applies the translation on his end
apply_translation = True
if apply_translation:
    mesh.apply_translation((0, 0, args.leica_z))

traj_vertices = trajectory.iloc[:, 0:3].to_numpy()
traj_segments = len(traj_vertices) - 1
entities = [trimesh.path.entities.Line(np.array([i, i + 1])) for i in range(traj_segments)] if traj_segments > 0 else []
trajectory_path = trimesh.path.Path3D(vertices=traj_vertices, entities=entities)
if traj_segments > 0:
    trajectory_path.colors = np.linspace(
        [0, 255, 0, 255],
        [255, 0, 0, 255],
        num=traj_segments,
        dtype=np.uint8,
    )
else:
    trajectory_path.colors = np.array([[0, 255, 0, 255]], dtype=np.uint8)

# mark the start and end points with green and red spheres # TODO could also do cones and use orientation aswell
start_sphere = trimesh.creation.icosphere(radius=0.1, subdivisions=3)
start_sphere.apply_translation(trajectory.iloc[0, 0:3].to_numpy())
start_sphere.visual.face_colors = [0, 255, 0, 255]
end_sphere = trimesh.creation.icosphere(radius=0.1, subdivisions=3)
end_sphere.apply_translation(trajectory.iloc[-1, 0:3].to_numpy())
end_sphere.visual.face_colors = [255, 0, 0, 255]

tube_radius = 1.32  # TODO tune
tube_offset = np.array([0.0, 0.0, -0.66])  # TODO tune
traj_points = trajectory.iloc[:, 0:3].to_numpy() + tube_offset
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

cut_z = False  # TODO this is buggy, don't use for now
if cut_z:
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
if not args.headless:
    scene.show(background=[131, 139, 150, 255])
mesh.export(output_path)
mtl_files = strip_obj_materials(output_path)
if mtl_files:
    cleanup_material_sidecars(output_path, mtl_files)
