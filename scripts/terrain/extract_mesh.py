import trimesh
import numpy as np
from pathlib import Path

# load data
data_path = Path("data/motions/raw/full_sequence-001.npz")
data = np.load(data_path, allow_pickle=True)

# extract terrain mesh
terrain_vertices = data["scene_vertices"]
terrain_faces = data["scene_faces"]
terrain_normals = data["scene_normals"]
terrain_vertex_colors = data["scene_vertex_colors"]

# create trimesh object
terrain_mesh = trimesh.Trimesh(
    vertices=terrain_vertices,
    faces=terrain_faces,
    vertex_normals=terrain_normals,
    vertex_colors=terrain_vertex_colors,
    process=False
)

terrain_mesh.invert()
terrain_mesh.export("data/terrains/raw/monday.obj")
terrain_mesh.show()
