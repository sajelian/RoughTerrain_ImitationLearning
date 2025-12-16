import trimesh

mesh = trimesh.load_mesh('source/RT_IL/RT_IL/assets/meshes/sequence_09_inv.obj')
#mesh.invert()

mesh.show(background=[131, 139, 150, 255])
