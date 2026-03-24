import torch
import numpy as np
import sys
from pathlib import Path

# 기본: 스크립트/레포지토리 기준의 상대 경로 사용, CLI로 파일 경로를 넘길 수도 있음
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    repo_root = Path(__file__).resolve().parents[2]  # Lidar-IMU-SLAM-DUNK
    workspace_root = repo_root.parent  # 상위 작업 폴더 (예: slamdunk)
    # 기본 파일은 워크스페이스 루트에 있는 pred_0.pth
    file_path = str(workspace_root / "pred_0.pth")

print(f"Loading {file_path}...")
try:
    data = torch.load(file_path, map_location="cpu")
    points = data["coord"].numpy()

    print(f"\n===== [데이터 진단 결과] =====")
    print(f"1. 점 개수: {len(points)}개 (60만개면 정상)")
    
    # 좌표 범위 확인
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    
    print(f"2. X 좌표 범위: {min_xyz[0]:.2f} ~ {max_xyz[0]:.2f}")
    print(f"3. Y 좌표 범위: {min_xyz[1]:.2f} ~ {max_xyz[1]:.2f}")
    print(f"4. Z 좌표 범위: {min_xyz[2]:.2f} ~ {max_xyz[2]:.2f}")

    # 크기 확인
    size = max_xyz - min_xyz
    print(f"5. 전체 크기(m): 가로 {size[0]:.2f}m / 세로 {size[1]:.2f}m / 높이 {size[2]:.2f}m")
    
    if np.isnan(points).any():
        print("🚨 경고: 좌표에 NaN(숫자 아님) 값이 포함되어 있습니다!")
    elif size[0] < 0.1 and size[1] < 0.1:
        print("⚠️ 경고: 데이터가 너무 작습니다! (밀리미터 단위일 수도 있음)")
    elif size[0] > 1000:
        print("⚠️ 경고: 데이터가 너무 큽니다! (좌표계 오류 가능성)")
    else:
        print("✅ 좌표 범위는 정상입니다. 뷰어에서 'Reset View'를 해보세요.")

except Exception as e:
    print(f"에러 발생: {e}")