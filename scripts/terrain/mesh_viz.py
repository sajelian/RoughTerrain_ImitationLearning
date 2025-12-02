import numpy as np
import trimesh

mesh = trimesh.load_mesh('source/RT_IL/RT_IL/assets/meshes/sequence_07.obj')
mesh.invert()
mesh.show()
