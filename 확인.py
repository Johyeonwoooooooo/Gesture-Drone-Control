import numpy as np
import glob
import os

root = "/Users/yoochaewon/Desktop/astar_rrt_project/npy"

folders = sorted(glob.glob(os.path.join(root, "*")))

if not folders:
    print(f"폴더를 찾지 못했습니다: {root}")
    print("경로를 확인해주세요!")
    exit()

print(f"총 {len(folders)}개 씬 발견\n")
print(f"{'씬 이름':<35} {'중심 X':>8} {'중심 Y':>8} {'중심 Z':>8} {'층':>5} {'포인트 수':>10}")
print("-" * 85)

scenes = []  # 정렬용 저장

for folder in folders:
    coord_path = os.path.join(folder, "coord.npy")
    if not os.path.exists(coord_path):
        print(f"[건너뜀] coord.npy 없음: {os.path.basename(folder)}")
        continue

    coords = np.load(coord_path)
    center = coords.mean(axis=0)
    z_mean = center[2]

    # 폴더 이름에서 층 추출 (예: 00809_Qpor2mEya8F_001_010 → 001)
    name = os.path.basename(folder)
    parts = name.split("_")
    floor_id = parts[-2] if len(parts) >= 2 else "?"  # 끝에서 두번째 = 층 번호

    # 폴더명 기반 층 + Z값 기반 층 둘 다 표시
    if floor_id in ("000", "001"):
        floor = "1층"
    elif floor_id == "002":
        floor = "2층"
    else:
        floor = "?"

    scenes.append({
        "name": name,
        "center": center,
        "floor": floor,
        "n_points": len(coords),
        "x_range": (coords[:,0].min(), coords[:,0].max()),
        "y_range": (coords[:,1].min(), coords[:,1].max()),
        "z_range": (coords[:,2].min(), coords[:,2].max()),
    })

    # 폴더 이름이 너무 기니까 짧게 표시
    short_name = name.replace("00809_Qpor2mEya8F_", "")
    print(f"{short_name:<35} {center[0]:>8.2f} {center[1]:>8.2f} {center[2]:>8.2f} {floor:>5} {len(coords):>10,}")

print(f"\n총 {len(scenes)}개 방 처리 완료\n")

# 추가 정보: 각 방의 좌표 범위 (start/goal 좌표 잡을 때 유용)
print("=" * 85)
print("[상세 좌표 범위] — A*/RRT* 시작-목표 좌표 잡을 때 참고")
print("=" * 85)
for s in scenes:  # 처음 5개만 샘플로 출력
    short = s["name"].replace("00809_Qpor2mEya8F_", "")
    print(f"\n{short} ({s['floor']}):")
    print(f"  X: {s['x_range'][0]:6.2f} ~ {s['x_range'][1]:6.2f}")
    print(f"  Y: {s['y_range'][0]:6.2f} ~ {s['y_range'][1]:6.2f}")
    print(f"  Z: {s['z_range'][0]:6.2f} ~ {s['z_range'][1]:6.2f}")

print("\n(나머지 방은 다 보고 싶으면 위 코드에서 scenes[:5] → scenes 로 변경)")