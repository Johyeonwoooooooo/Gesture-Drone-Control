import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("playground/3D/3d_scan.ply")
print(f"원본: {len(pcd.points)}개")

bbox = pcd.get_axis_aligned_bounding_box()
print(f"Bounding Box: {bbox.get_extent()}")

# 5cm voxel은 적당 - 유지
voxel_size = 0.05
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"다운샘플 완료: {len(pcd.points)}개")

# 이상치 제거 꼭 켜기
pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
pcd = pcd.select_by_index(ind)
print(f"이상치 제거 후: {len(pcd.points)}개")

# radius를 voxel_size * 3 = 0.15로, max_nn도 늘리기
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=50)
)

# 89만개면 orient k=30으로 낮춰야 속도 빠름
pcd.orient_normals_consistent_tangent_plane(k=30)
print("법선 정렬 완료!")

# 89만개 포인트 + 8m 공간 → depth=10이 적합
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=10, width=0, scale=1.1, linear_fit=False
)
print(f"Mesh 버텍스: {len(mesh.vertices)}개")

# 외곽 아티팩트 제거 - 0.05~0.1 사이 조절
densities = np.asarray(densities)
threshold = np.quantile(densities, 0.05)
mesh.remove_vertices_by_mask(densities < threshold)

mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh("playground/3D/output.obj", mesh)
print("저장 완료!")