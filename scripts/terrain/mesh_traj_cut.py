import numpy as np
import trimesh
from math import pi

# TODO instead of cutting out a box, we cut out around the trajectory path (see drawing on my mirror)
# TODO also think about reducing the number of vertices by simplifying the mesh
# we could use the same idea from above but only reduce scale in a slimmer tube around the path

mesh = trimesh.load_mesh('source/RT_IL/RT_IL/assets/meshes/sequence_07.obj')
mesh.invert()
# cut out a cubeoid of the mesh
rotation = trimesh.transformations.rotation_matrix(angle=-pi/12, direction=[0, 0, 1], point=[0, 0, 0])
translation = trimesh.transformations.translation_matrix([-1.0, -1.5, -2.0])
box = trimesh.creation.box(extents=[2.0, 2.0, 0.45], transform=translation + rotation)
box.apply_translation([0.0, 0.0, 1.0])
obb = box.bounding_box_oriented
obb_transform = obb.primitive.transform if hasattr(obb, "primitive") else obb.transform
half_extents = (obb.primitive.extents if hasattr(obb, "primitive") else obb.extents) / 2.0
world_to_box = np.linalg.inv(obb_transform)
mesh_in_box_frame = mesh.copy()
mesh_in_box_frame.apply_transform(world_to_box)
inside_vertices = np.all(np.abs(mesh_in_box_frame.vertices) <= (half_extents + 1e-8), axis=1)
faces_mask = inside_vertices[mesh_in_box_frame.faces].all(axis=1)
mesh = mesh.submesh([faces_mask], append=True, repair=False) if faces_mask.any() else trimesh.Trimesh()
# mesh = trimesh.util.concatenate([mesh, box])

mesh.show()  # type: ignore
mesh.export('source/RT_IL/RT_IL/assets/meshes/sequence_07_fix.obj') # type: ignore
