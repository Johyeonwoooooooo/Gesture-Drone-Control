import open3d as o3d

mesh = o3d.io.read_triangle_mesh("playground/3D/output.obj")
print(f"버텍스 수: {len(mesh.vertices)}")
print(f"삼각형 수: {len(mesh.triangles)}")
o3d.visualization.draw_geometries([mesh])  # 뷰어 창 열림