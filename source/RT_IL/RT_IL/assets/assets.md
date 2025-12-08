### USD Assets
- This directory contains USD assets aswell as the robot description.
- Each usd-file has a corresponding configuration yaml that is used during the asset loading process.

To generate a new USD asset from a mesh file (e.g., .stl, .obj), you can use the following command:
`python ~/IsaacLab/scripts/tools/convert_mesh.py source/RT_IL/RT_IL/assets/meshes/<mesh_name>.obj source/RT_IL/RT_IL/assets/usd/<mesh_name>/<mesh_name>.usd --make-instanceable --collision-approximation sdf`

In `config.yaml` you can specify translation, rotation and scaling of the mesh asset during loading.