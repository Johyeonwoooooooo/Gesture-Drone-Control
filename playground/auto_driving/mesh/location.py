import numpy as np
import glob
import os

# ← 여기 경로만 본인 경로로 바꿔줘
root = "playground\\auto_driving\\compressed_npy"

folders = sorted(glob.glob(os.path.join(root, "*")))

if not folders:
    print(f"폴더를 찾지 못했습니다: {root}")
    print("경로를 확인해주세요!")
    input("엔터 누르면 종료...")
    exit()

print(f"총 {len(folders)}개 씬 발견\n")
print(f"{'씬 이름':<45} {'중심 X':>8} {'중심 Y':>8} {'중심 Z':>8} {'층':>5} {'포인트 수':>10}")
print("-" * 95)

for folder in folders:
    coord_path = os.path.join(folder, "coord.npy")
    if not os.path.exists(coord_path):
        continue

    coords = np.load(coord_path)
    center = coords.mean(axis=0)
    z_mean = center[2]

    # Z값으로 층 추정 (데이터셋마다 다를 수 있음)
    if z_mean < 1.5:
        floor = "1층"
    else:
        floor = "2층"

    name = os.path.basename(folder)
    print(f"{name:<45} {center[0]:>8.2f} {center[1]:>8.2f} {center[2]:>8.2f} {floor:>5} {len(coords):>10,}")

print("\n완료!")
input("엔터 누르면 종료...")