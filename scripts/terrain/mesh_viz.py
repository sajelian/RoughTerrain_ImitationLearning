import trimesh

mesh = trimesh.load_mesh('source/RT_IL/RT_IL/assets/meshes/sequence_07.obj')
mesh.invert()  #TODO remove this if Alex fixes the mesh orientation on his end
mesh.show()
