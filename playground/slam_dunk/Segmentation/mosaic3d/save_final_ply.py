import torch
import numpy as np
import open3d as o3d
import os
from pathlib import Path
import sys

# 파일 경로는 CLI 인자로 받거나, 스크립트/워크스페이스 기준 상대경로를 기본값으로 사용합니다.
if len(sys.argv) > 1:
	file_path = sys.argv[1]
else:
	repo_root = Path(__file__).resolve().parents[2]  # Lidar-IMU-SLAM-DUNK
	workspace_root = repo_root.parent
	# 기본적으로 워크스페이스 루트에 있는 pred_0.pth 사용
	file_path = str(workspace_root / "pred_0.pth")

print(f"Loading {file_path}...")
data = torch.load(file_path, map_location="cpu")
points = data["coord"].numpy()
preds = data["pred"].numpy()

print(f"Points: {len(points)}")

# 1. 클래스별 랜덤 색상 생성
num_classes = preds.max() + 1
colormap = np.random.rand(num_classes + 1, 3)
semantic_colors = colormap[preds]

# 2. Point Cloud 만들기
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(semantic_colors)

# 3. 파일로 저장 (창 안 띄움)
save_name = "final_result.ply"
o3d.io.write_point_cloud(save_name, pcd)
print(f"\n✅ [저장 완료] {os.path.abspath(save_name)}")
print("👉 이 파일을 로컬 컴퓨터(내 PC)로 다운로드해서 열어보세요!")