#!/usr/bin/env python3
"""
방별 npy → GLB 3D 시각화
+ A*/RRT* 경로 + waypoint별 방 라벨링

각 waypoint에:
  - 방별 고유 색상 (어느 방을 지나는지 한눈에)
  - 세로 막대(pole)로 위치 강조
  - GLB 노드 이름에 정보 ("astar_wp_03_in_001_006")
별도 출력:
  - 콘솔 표
  - path_summary.txt
"""

from __future__ import annotations

import argparse
import colorsys
import glob
import importlib.util
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import trimesh


# ──────────────────────────────────────────────────────────────────────────────
# 3D.py 동적 import
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
PATHFINDER_PATH = SCRIPT_DIR / "3D.py"

spec = importlib.util.spec_from_file_location("pathfinder", PATHFINDER_PATH)
pathfinder = importlib.util.module_from_spec(spec)
sys.modules["pathfinder"] = pathfinder
spec.loader.exec_module(pathfinder)

voxelize    = pathfinder.voxelize
astar       = pathfinder.astar
rrt_star    = pathfinder.rrt_star
smooth_path = pathfinder.smooth_path
path_length = pathfinder.path_length


# ──────────────────────────────────────────────────────────────────────────────
# 방 로드 (워커)
# ──────────────────────────────────────────────────────────────────────────────
def load_one_room(args):
    folder, max_points = args
    name = os.path.basename(folder).replace("00809_Qpor2mEya8F_", "")
    coord_path = os.path.join(folder, "coord.npy")
    color_path = os.path.join(folder, "color.npy")

    if not os.path.exists(coord_path):
        return None

    coords = np.load(coord_path).astype(np.float32)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 3)
    coords = coords[:, :3]

    if os.path.exists(color_path):
        colors = np.load(color_path).astype(np.float32)
        if colors.max() > 1.5:
            colors = colors / 255.0
        colors = np.clip(colors[:, :3], 0, 1)
        colors = (colors * 255).astype(np.uint8)
        alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
        colors = np.hstack([colors, alpha])
    else:
        group = name.split("_")[0]
        fallback = {"000": [92, 156, 214, 255],
                    "001": [112, 173, 71, 255],
                    "002": [193, 154, 107, 255]}.get(group, [128, 128, 128, 255])
        colors = np.tile(fallback, (len(coords), 1)).astype(np.uint8)

    if len(coords) > max_points:
        idx = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[idx]
        colors = colors[idx]

    return {
        "name": name,
        "coords": coords,
        "colors": colors,
        "xyz_min": coords.min(axis=0),
        "xyz_max": coords.max(axis=0),
        "center":  coords.mean(axis=0),
    }


def load_all_rooms_parallel(npy_dir, max_points_per_room, n_workers):
    folders = sorted(glob.glob(str(npy_dir / "*")))
    folders = [f for f in folders if os.path.isdir(f)]
    if not folders:
        return None, None, []

    job_args = [(f, max_points_per_room) for f in folders]
    print(f"[Load] 방 {len(folders)}개 병렬 로드 ({n_workers} 워커)...")
    t0 = time.perf_counter()
    with Pool(processes=n_workers) as pool:
        rooms = [r for r in pool.map(load_one_room, job_args) if r is not None]
    print(f"  → {time.perf_counter()-t0:.2f}초")

    rooms.sort(key=lambda r: r["name"])
    all_coords = np.vstack([r["coords"] for r in rooms])
    all_colors = np.vstack([r["colors"] for r in rooms])
    return all_coords, all_colors, rooms


def find_room_containing(rooms, point):
    """좌표가 어느 방 bbox 안에 있는지 (가장 가까운 방으로 fallback)"""
    p = np.array(point)
    for r in rooms:
        if np.all(r["xyz_min"] <= p) and np.all(p <= r["xyz_max"]):
            return r
    # bbox 밖이면 가장 가까운 방의 중심으로
    best = None; best_d = float('inf')
    for r in rooms:
        d = float(np.linalg.norm(r["center"] - p))
        if d < best_d:
            best_d = d; best = r
    return best


def assign_room_colors(rooms):
    """방마다 고유 색상 (HSV 색상환 균등 분할)"""
    n = len(rooms)
    color_map = {}
    for i, r in enumerate(rooms):
        h = i / n
        rgb = colorsys.hsv_to_rgb(h, 0.85, 0.95)
        color_map[r["name"]] = (
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255), 255
        )
    return color_map


# ──────────────────────────────────────────────────────────────────────────────
# 경로 계획
# ──────────────────────────────────────────────────────────────────────────────
def plan_paths(coords, start, goal, resolution, sample):
    print(f"\n[Voxelize] resolution={resolution}m, sample=1/{sample}")
    t0 = time.perf_counter()
    gm = voxelize(coords.astype(float), resolution=resolution,
                  margin=0, sample=sample)
    print(f"  Grid shape: {gm.shape}, free: {gm.free_ratio*100:.1f}%, "
          f"({time.perf_counter()-t0:.2f}초)")

    print(f"\n[A*] 경로 계산 중...")
    t0 = time.perf_counter()
    a_path, _ = astar(gm, np.array(start), np.array(goal))
    a_time = (time.perf_counter() - t0) * 1000
    if a_path is not None:
        a_path = smooth_path(a_path, gm)
        print(f"  ✅ 성공: {a_time:.1f}ms, 거리 {path_length(a_path):.2f}m, "
              f"waypoint {len(a_path)}개")
    else:
        print(f"  ❌ 실패: {a_time:.1f}ms")

    print(f"\n[RRT*] 경로 계산 중...")
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()
    r_path, _ = rrt_star(gm, np.array(start), np.array(goal), rng,
                         max_iter=3000, step_len=0.5,
                         rewire_radius=1.5, goal_bias=0.1)
    r_time = (time.perf_counter() - t0) * 1000
    if r_path is not None:
        print(f"  ✅ 성공: {r_time:.1f}ms, 거리 {path_length(r_path):.2f}m, "
              f"waypoint {len(r_path)}개")
    else:
        print(f"  ❌ 실패: {r_time:.1f}ms")

    return a_path, r_path


# ──────────────────────────────────────────────────────────────────────────────
# 3D 메시 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def make_tube_segment(p1, p2, radius, color, sections=6):
    """두 점 사이를 잇는 cylinder"""
    seg_len = float(np.linalg.norm(p2 - p1))
    if seg_len < 1e-6:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=seg_len,
                                    sections=sections)
    direction = (p2 - p1) / seg_len
    z_axis = np.array([0, 0, 1])
    if not np.allclose(direction, z_axis):
        rot = trimesh.geometry.align_vectors(z_axis, direction)
        cyl.apply_transform(rot)
    cyl.apply_translation((p1 + p2) / 2.0)
    cyl.visual.face_colors = np.tile(color, (len(cyl.faces), 1))
    return cyl


def make_pole(position, height=1.5, radius=0.02, color=(200, 200, 200, 200)):
    """waypoint 위치에 세로 막대 (위치 강조)"""
    cyl = trimesh.creation.cylinder(radius=radius, height=height, sections=6)
    cyl.apply_translation([position[0], position[1], position[2] + height/2])
    cyl.visual.face_colors = np.tile(color, (len(cyl.faces), 1))
    return cyl


def make_sphere(position, radius=0.12, color=(0, 148, 212, 255)):
    s = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    s.apply_translation(position)
    s.visual.face_colors = np.tile(color, (len(s.faces), 1))
    return s


# ──────────────────────────────────────────────────────────────────────────────
# Waypoint 라벨링 (어느 방인지 식별 + 요약)
# ──────────────────────────────────────────────────────────────────────────────
def label_waypoints(path, rooms):
    """각 waypoint마다 (waypoint_idx, 좌표, 방 이름, 누적거리) 정보"""
    if path is None:
        return []
    labels = []
    cum_dist = 0.0
    for i, wp in enumerate(path):
        if i > 0:
            cum_dist += float(np.linalg.norm(np.array(wp) - np.array(path[i-1])))
        room = find_room_containing(rooms, wp)
        labels.append({
            "idx": i,
            "pos": tuple(float(x) for x in wp),
            "room": room["name"] if room else "(unknown)",
            "cum_dist": cum_dist,
        })
    return labels


def print_waypoint_table(label_list, algo_name):
    if not label_list:
        print(f"\n[{algo_name}] 경로 없음"); return

    print(f"\n{'='*80}")
    print(f"  {algo_name} 경로 — Waypoint별 통과 방")
    print(f"{'='*80}")
    print(f"{'#':>3} {'X':>8} {'Y':>8} {'Z':>8}  {'방':<12} {'누적거리':>10}")
    print(f"{'-'*80}")
    prev_room = None
    for lb in label_list:
        marker = ""
        if lb["room"] != prev_room:
            marker = "  ← 방 전환"
            prev_room = lb["room"]
        print(f"{lb['idx']:>3} {lb['pos'][0]:>8.2f} {lb['pos'][1]:>8.2f} "
              f"{lb['pos'][2]:>8.2f}  {lb['room']:<12} "
              f"{lb['cum_dist']:>8.2f}m{marker}")
    print(f"{'='*80}")

    # 통과한 방 시퀀스
    seq = []
    for lb in label_list:
        if not seq or seq[-1] != lb["room"]:
            seq.append(lb["room"])
    print(f"  통과한 방 순서: {' → '.join(seq)}")
    print(f"  총 방 전환 횟수: {len(seq) - 1}회")


def save_summary_txt(a_labels, r_labels, start, goal, output_path):
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("  경로 계획 결과 요약\n")
        f.write("=" * 80 + "\n")
        f.write(f"  전역 시작: {start}\n")
        f.write(f"  전역 목표: {goal}\n\n")

        for name, labels in [("A*", a_labels), ("RRT*", r_labels)]:
            f.write(f"\n[{name}]\n")
            f.write("-" * 80 + "\n")
            if not labels:
                f.write("  실패\n"); continue
            f.write(f"{'#':>3} {'X':>8} {'Y':>8} {'Z':>8}  {'방':<14} "
                    f"{'누적거리':>10}\n")
            for lb in labels:
                f.write(f"{lb['idx']:>3} {lb['pos'][0]:>8.2f} "
                        f"{lb['pos'][1]:>8.2f} {lb['pos'][2]:>8.2f}  "
                        f"{lb['room']:<14} {lb['cum_dist']:>8.2f}m\n")
            # 방 시퀀스
            seq = []
            for lb in labels:
                if not seq or seq[-1] != lb["room"]:
                    seq.append(lb["room"])
            f.write(f"\n  통과한 방: {' → '.join(seq)}\n")
            f.write(f"  방 전환: {len(seq)-1}회, 총 거리: "
                    f"{labels[-1]['cum_dist']:.2f}m\n")
    print(f"  → 요약 저장: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# GLB 빌드
# ──────────────────────────────────────────────────────────────────────────────
def build_glb(coords, colors, a_path, r_path, start, goal,
              rooms, room_color_map, a_labels, r_labels, output_glb):
    scene = trimesh.Scene()

    # 1. 점군
    print(f"\n[GLB] 점군 추가 ({len(coords):,}점)")
    pc = trimesh.PointCloud(vertices=coords, colors=colors)
    scene.add_geometry(pc, geom_name="point_cloud")

    # 2. A* 경로: 세그먼트별로 통과 방 색상
    if a_path is not None and a_labels:
        print(f"[GLB] A* 경로 ({len(a_path)} waypoint)")
        for i in range(len(a_path) - 1):
            p1, p2 = np.array(a_path[i]), np.array(a_path[i+1])
            mid = (p1 + p2) / 2.0
            mid_room = find_room_containing(rooms, mid)
            seg_color = room_color_map.get(mid_room["name"],
                                           (0, 148, 212, 255)) \
                        if mid_room else (0, 148, 212, 255)
            tube = make_tube_segment(p1, p2, radius=0.06, color=seg_color)
            if tube is not None:
                scene.add_geometry(
                    tube,
                    geom_name=f"astar_seg_{i:02d}_through_"
                              f"{mid_room['name'] if mid_room else 'unknown'}"
                )

        # waypoint 마커 + pole
        for lb in a_labels:
            room_color = room_color_map.get(lb["room"], (0, 148, 212, 255))
            wp_pos = lb["pos"]

            # waypoint 구
            sph = make_sphere(wp_pos, radius=0.15, color=room_color)
            scene.add_geometry(
                sph,
                geom_name=f"astar_wp_{lb['idx']:02d}_in_{lb['room']}"
            )

            # 세로 막대 (위치 강조)
            pole = make_pole(wp_pos, height=1.0, radius=0.02,
                             color=(*room_color[:3], 180))
            scene.add_geometry(
                pole,
                geom_name=f"astar_pole_{lb['idx']:02d}_in_{lb['room']}"
            )

    # 3. RRT* 경로 (보라색 단일, 비교용)
    if r_path is not None and r_labels:
        print(f"[GLB] RRT* 경로 ({len(r_path)} waypoint)")
        for i in range(len(r_path) - 1):
            p1, p2 = np.array(r_path[i]), np.array(r_path[i+1])
            tube = make_tube_segment(p1, p2, radius=0.04,
                                     color=(124, 58, 237, 200))
            if tube is not None:
                scene.add_geometry(tube, geom_name=f"rrt_seg_{i:02d}")

        # RRT* waypoint (작게)
        for lb in r_labels:
            sph = make_sphere(lb["pos"], radius=0.08,
                              color=(124, 58, 237, 255))
            scene.add_geometry(
                sph,
                geom_name=f"rrt_wp_{lb['idx']:02d}_in_{lb['room']}"
            )

    # 4. 시작/목표 큰 마커
    print(f"[GLB] 시작/목표 마커")
    start_sphere = make_sphere(start, radius=0.35, color=(50, 255, 50, 255))
    goal_sphere  = make_sphere(goal,  radius=0.35, color=(255, 30, 30, 255))
    scene.add_geometry(start_sphere, geom_name="START")
    scene.add_geometry(goal_sphere,  geom_name="GOAL")

    # 5. 저장
    print(f"\n[GLB] 저장 중: {output_glb}")
    scene.export(str(output_glb))
    print(f"  파일 크기: {os.path.getsize(output_glb)/1024/1024:.2f} MB")


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-dir", type=Path, default=Path("npy"))
    parser.add_argument("--global-start", type=float, nargs=3,
                        default=[1.0, 6.0, 1.5])
    parser.add_argument("--global-goal", type=float, nargs=3,
                        default=[13.0, -3.0, 4.5])
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--resolution", type=float, default=0.15)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--max-points-per-room", type=int, default=6000)
    parser.add_argument("--output", type=Path,
                        default=Path("path_3d_labeled.glb"))
    parser.add_argument("--summary", type=Path,
                        default=Path("path_summary.txt"))
    args = parser.parse_args()

    print("=" * 60)
    print("  3D 점군 + A*/RRT* 경로 + Waypoint 라벨링 → GLB")
    print("=" * 60)
    print(f"  npy 폴더:     {args.npy_dir}")
    print(f"  병렬 워커:    {args.workers}")
    print(f"  전역 시작:    {tuple(args.global_start)}")
    print(f"  전역 목표:    {tuple(args.global_goal)}")
    print("=" * 60)

    coords, colors, rooms = load_all_rooms_parallel(
        args.npy_dir, args.max_points_per_room, args.workers)
    if coords is None:
        print(f"[에러] 방 못 찾음: {args.npy_dir}"); return
    print(f"  방 {len(rooms)}개, 총 {len(coords):,}점")

    # 방별 색상 맵
    room_color_map = assign_room_colors(rooms)
    print(f"  방별 색상 맵 생성 완료 ({len(room_color_map)}색)")

    start_room = find_room_containing(rooms, args.global_start)
    goal_room  = find_room_containing(rooms, args.global_goal)
    print(f"  시작 방: {start_room['name'] if start_room else '(없음)'}")
    print(f"  목표 방: {goal_room['name']  if goal_room  else '(없음)'}")

    # 경로 계획
    a_path, r_path = plan_paths(coords, args.global_start, args.global_goal,
                                args.resolution, args.sample)

    # waypoint 라벨링
    a_labels = label_waypoints(a_path, rooms)
    r_labels = label_waypoints(r_path, rooms)

    # 콘솔 표 출력
    print_waypoint_table(a_labels, "A*")
    print_waypoint_table(r_labels, "RRT*")

    # 텍스트 요약 저장
    print(f"\n[Save] 요약 텍스트")
    save_summary_txt(a_labels, r_labels,
                     tuple(args.global_start), tuple(args.global_goal),
                     args.summary)

    # GLB 빌드
    build_glb(coords, colors, a_path, r_path,
              args.global_start, args.global_goal,
              rooms, room_color_map, a_labels, r_labels, args.output)

    print(f"\n✅ 완료")
    print(f"  GLB:     {args.output.resolve()}")
    print(f"  Summary: {args.summary.resolve()}")
    print(f"\n[GLB 열기]")
    print(f"  • https://gltf-viewer.donmccurdy.com/  (드래그&드롭)")
    print(f"    → 'Show Scene Hierarchy' 켜면 노드 이름으로 어느 waypoint가")
    print(f"      어느 방에 있는지 확인 가능!")
    print(f"  • macOS: 파일 클릭 후 스페이스바 (Quick Look)")


if __name__ == "__main__":
    main()