import open3d as o3d
import numpy as np
import os

input_path = "playground/3D/3d_scan.ply"
output_path = "playground/tello_simulator/Assets/v3_house_scan.obj"

if not os.path.exists(input_path):
    print(f"Error: {input_path} not found.")
    exit(1)

print(f"Loading {input_path}...")
pcd = o3d.io.read_point_cloud(input_path)
# pcd = pcd.voxel_down_sample(voxel_size=0.02)
print(f"Downsampled: {len(pcd.points)} points")

# Estimate normals
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=50)
)
print("Normals estimated")

pcd.orient_normals_consistent_tangent_plane(k=30)
print("Normals oriented")

# Poisson reconstruction
print("Running Poisson reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9, width=0, scale=1.1, linear_fit=False
)
print(f"Mesh created with {len(mesh.vertices)} vertices")

# Transfer colors from point cloud to mesh vertices
if pcd.has_colors():
    print("Transferring colors...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_colors = []
    for vertex in mesh.vertices:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(vertex, 1)
        mesh_colors.append(pcd.colors[idx[0]])
    mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

# Remove low density vertices (noise)
densities = np.asarray(densities)
threshold = np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(densities < threshold)

mesh.compute_vertex_normals()
print(f"Saving to {output_path}...")
o3d.io.write_triangle_mesh(output_path, mesh)
print("Done! You can now find house_scan.obj in your Unity Assets folder.")