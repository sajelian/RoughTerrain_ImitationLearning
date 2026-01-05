import trimesh

mesh = trimesh.load_mesh('source/RT_IL/RT_IL/assets/meshes/terrain.obj')
#mesh.invert()

mesh.show(background=[131, 139, 150, 255])
