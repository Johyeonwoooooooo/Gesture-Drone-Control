import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("playground/3D/3d_scan.ply")
pcd = pcd.voxel_down_sample(voxel_size=0.05)
print(f"다운샘플 완료: {len(pcd.points)}개")

# outlier 제거 완전 스킵
# 노말 추정
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=50)
)
print("노말 추정 완료")

pcd.orient_normals_consistent_tangent_plane(k=30)
print("노말 정렬 완료")

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9, width=0, scale=1.1, linear_fit=False
)
print(f"Mesh 버텍스: {len(mesh.vertices)}개")

densities = np.asarray(densities)
threshold = np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(densities < threshold)

mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("playground/3D/output.obj", mesh)
print("저장 완료!")