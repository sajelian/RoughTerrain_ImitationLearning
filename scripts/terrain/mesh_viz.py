import trimesh

mesh = trimesh.load_mesh('data/terrains/raw/lee_h.obj')
#mesh.invert()

mesh.show(background=[131, 139, 150, 255])
