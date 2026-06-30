#!/usr/bin/env python3
"""방별 npy 파일을 병렬로 A*/RRT* 처리하는 벤치마크.
rooms_graph.json을 활용해서 실제 통로(passages) 위치를 반영."""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# 3D.py 동적 import
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
PATHFINDER_PATH = SCRIPT_DIR / "3D.py"

spec = importlib.util.spec_from_file_location("pathfinder", PATHFINDER_PATH)
pathfinder = importlib.util.module_from_spec(spec)
sys.modules["pathfinder"] = pathfinder
spec.loader.exec_module(pathfinder)

voxelize = pathfinder.voxelize
astar = pathfinder.astar
rrt_star = pathfinder.rrt_star
smooth_path = pathfinder.smooth_path
path_length = pathfinder.path_length


# ──────────────────────────────────────────────────────────────────────────────
# JSON 로드 및 메타데이터 구성
# ──────────────────────────────────────────────────────────────────────────────

def load_rooms_metadata(json_path):
    """rooms_graph.json 로드 및 방-별 메타데이터 구성."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        rooms_meta = {}
        for room_id, room_info in data.get("rooms", {}).items():
            rooms_meta[room_id] = {
                "bbox_min": np.array(room_info["bbox_min"], dtype=float),
                "bbox_max": np.array(room_info["bbox_max"], dtype=float),
                "center": np.array(room_info["center"], dtype=float),
                "passages": [np.array(p, dtype=float) for p in room_info.get("passages", [])],
                "npy": room_info["npy"],
            }

        # room_id → npy_name 매핑
        npy_to_room = {v["npy"]: k for k, v in rooms_meta.items()}

        return rooms_meta, npy_to_room
    except Exception as e:
        print(f"[경고] JSON 로드 실패 ({json_path}): {e}")
        return {}, {}


def find_room_by_point(point, rooms_meta):
    """point가 어느 방에 속하는지 찾기."""
    p = np.array(point)
    for room_id, meta in rooms_meta.items():
        if np.all(meta["bbox_min"] <= p) and np.all(p <= meta["bbox_max"]):
            return room_id
    return None


def get_start_goal_for_room(room_id, rooms_meta, passages_mode=True):
    """해당 방에서 시작/끝 좌표 결정.
    passages_mode=True면 passages 위치를 활용, False면 bbox 기반."""
    meta = rooms_meta.get(room_id)
    if not meta:
        return None, None

    bbox_min, bbox_max = meta["bbox_min"], meta["bbox_max"]
    passages = meta["passages"]

    # ① 시작점: passages가 있으면 첫 번째 passage에서 room 내부로 조금 들어간 곳
    if passages_mode and len(passages) > 0:
        start = passages[0] + (meta["center"] - passages[0]) * 0.2
    else:
        # bbox 안쪽으로 10% 여유
        delta = (bbox_max - bbox_min) * 0.1
        start = bbox_min + delta

    # ② 끝점: passages가 있으면 마지막 passage에서 room 내부로 조금 들어간 곳
    if passages_mode and len(passages) > 0:
        goal = passages[-1] + (meta["center"] - passages[-1]) * 0.2
    else:
        delta = (bbox_max - bbox_min) * 0.1
        goal = bbox_max - delta

    return start, goal


# ──────────────────────────────────────────────────────────────────────────────
# 한 방 처리
# ──────────────────────────────────────────────────────────────────────────────

def process_room(args):
    (folder_path, resolution, sample, margin,
     rrt_iter, rrt_step, rrt_radius, rrt_bias, seed,
     global_start, global_goal, rooms_meta, npy_to_room) = args  # ✨ rooms_meta, npy_to_room 추가

    name = os.path.basename(folder_path)
    coord_path = os.path.join(folder_path, "coord.npy")

    if not os.path.exists(coord_path):
        return {"room": name, "error": "coord.npy 없음"}

    try:
        points = np.load(coord_path).astype(float)
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        points = points[:, :3]
        xyz_min = points.min(axis=0)
        xyz_max = points.max(axis=0)

        gs = np.array(global_start)
        gg = np.array(global_goal)
        contains_start = bool(np.all(xyz_min <= gs) and np.all(gs <= xyz_max))
        contains_goal = bool(np.all(xyz_min <= gg) and np.all(gg <= xyz_max))

        # ✨ [수정] rooms_meta 활용해서 더 정확한 start/goal 결정
        room_id = npy_to_room.get(name)
        if room_id and room_id in rooms_meta:
            meta = rooms_meta[room_id]
            # 이 방이 시작 방인 경우
            if contains_start:
                start = gs.copy()
                # passages 중 global_goal 방향의 가장 가까운 passage 찾기
                if meta["passages"]:
                    passages = meta["passages"]
                    dists = [np.linalg.norm(p - gg) for p in passages]
                    closest_passage = passages[np.argmin(dists)]
                    goal = closest_passage.copy()
                else:
                    goal = meta["center"].copy()
            # 이 방이 끝 방인 경우
            elif contains_goal:
                goal = gg.copy()
                # passages 중 global_start 방향의 가장 가까운 passage 찾기
                if meta["passages"]:
                    passages = meta["passages"]
                    dists = [np.linalg.norm(p - gs) for p in passages]
                    closest_passage = passages[np.argmin(dists)]
                    start = closest_passage.copy()
                else:
                    start = meta["center"].copy()
            # 중간 방인 경우 (양쪽 다 포함 안 함)
            else:
                if len(meta["passages"]) >= 2:
                    # passages 중 첫 번째와 마지막 활용
                    start = meta["passages"][0].copy()
                    goal = meta["passages"][-1].copy()
                else:
                    # 방의 중심을 중심으로 시작/끝점 설정
                    delta = (xyz_max - xyz_min) * 0.1
                    start = xyz_min + delta
                    goal = xyz_max - delta
        else:
            # rooms_meta 없는 경우 fallback
            delta = (xyz_max - xyz_min) * 0.1
            start = xyz_min + delta
            goal = xyz_max - delta

        gm = voxelize(points, resolution, margin, sample)

        # A*
        t0 = time.perf_counter()
        a_path, a_info = astar(gm, start, goal)
        a_time = (time.perf_counter() - t0) * 1000.0
        if a_path is not None:
            a_path = smooth_path(a_path, gm)

        # RRT*
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        r_path, r_info = rrt_star(gm, start, goal, rng,
                                  rrt_iter, rrt_step, rrt_radius, rrt_bias)
        r_time = (time.perf_counter() - t0) * 1000.0

        return {
            "room": name,
            "room_id": room_id,  # ✨ [추가]
            "n_points": len(points),
            "diagonal_m": float(np.linalg.norm(goal - start)),
            "contains_global_start": contains_start,
            "contains_global_goal": contains_goal,

            "astar_success": a_path is not None,
            "astar_time_ms": a_time,
            "astar_distance": path_length(a_path) if a_path else None,
            "astar_waypoints": len(a_path) if a_path else 0,

            "rrt_success": r_path is not None,
            "rrt_time_ms": r_time,
            "rrt_distance": path_length(r_path) if r_path else None,
            "rrt_waypoints": len(r_path) if r_path else 0,

            "error": None,
        }
    except Exception as e:
        return {"room": name, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────────────────────────────────────

def print_table(results, par_time, n_workers, seq_time=None):
    print("\n" + "=" * 130)
    print(f"{'Room':<20} {'ID':<6} {'대각선':>8} {'A* 시간':>11} {'A* 거리':>9} "
          f"{'RRT* 시간':>13} {'RRT* 거리':>11} {'전역점':>8}")
    print("-" * 130)

    a_times, r_times = [], []
    start_room, goal_room = None, None

    for r in results:
        if r.get("error"):
            short = r["room"].replace("00809_Qpor2mEya8F_", "")
            print(f"{short:<20}  [에러] {r['error']}")
            continue

        short = r["room"].replace("00809_Qpor2mEya8F_", "")
        room_id = r.get("room_id", "?")
        diag = r["diagonal_m"]
        a_t = f"{r['astar_time_ms']:>10.1f}" if r["astar_success"] else "      실패"
        a_d = f"{r['astar_distance']:>8.2f}" if r["astar_success"] else "       -"
        r_t = f"{r['rrt_time_ms']:>12.1f}" if r["rrt_success"] else "        실패"
        r_d = f"{r['rrt_distance']:>10.2f}" if r["rrt_success"] else "         -"

        global_marker = ""
        if r["contains_global_start"]:
            global_marker = "🟢시작"
            start_room = f"{room_id}({short})"
        if r["contains_global_goal"]:
            global_marker = "🔴목표"
            goal_room = f"{room_id}({short})"

        print(f"{short:<20} {room_id:<6} {diag:>6.2f}m {a_t}ms {a_d}m {r_t}ms {r_d}m {global_marker:>8}")

        if r["astar_success"]: a_times.append(r["astar_time_ms"])
        if r["rrt_success"]:   r_times.append(r["rrt_time_ms"])

    print("=" * 130)

    print(f"\n[경로 분석]")  # ✨ [수정] 레이블
    print(f"  시작점 위치:   {start_room or '(어느 방에도 안 속함)'}")
    print(f"  목표점 위치:   {goal_room or '(어느 방에도 안 속함)'}")

    if a_times:
        seq_calc = sum(a_times) + sum(r_times)
        print(f"\n[A*]   성공 {len(a_times):2d}개, 평균 {np.mean(a_times):>8.1f}ms, 최대 {max(a_times):>8.1f}ms")
        print(f"[RRT*] 성공 {len(r_times):2d}개, 평균 {np.mean(r_times):>8.1f}ms, 최대 {max(r_times):>8.1f}ms")

        print(f"\n{'─' * 70}")
        print(f"  ① 환산 순차 시간  (방 시간 단순 합):        {seq_calc / 1000:>8.2f}초")
        if seq_time is not None:
            print(f"  ② 실제 순차 시간  (for loop 실측):          {seq_time:>8.2f}초")
        print(f"  ③ 실제 병렬 시간  ({n_workers:2d} 워커, 실측):       {par_time:>8.2f}초")
        print(f"{'─' * 70}")
        print(f"  환산 기준 가속비  (① / ③):                  {seq_calc / 1000 / par_time:>8.2f}x")
        if seq_time is not None:
            print(f"  실측 기준 가속비  (② / ③):                  {seq_time / par_time:>8.2f}x  ⭐")
        print(f"{'─' * 70}")


def plot_results(results, output, par_time, n_workers, seq_time=None):
    valid = [r for r in results if not r.get("error") and r["astar_success"]]
    if not valid:
        print("그래프 그릴 데이터 없음");
        return

    names = [r["room"].replace("00809_Qpor2mEya8F_", "") for r in valid]
    a_times = [r["astar_time_ms"] for r in valid]
    r_times = [r["rrt_time_ms"] if r["rrt_success"] else 0 for r in valid]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    x = np.arange(len(names))
    w = 0.4

    ax1.bar(x - w / 2, a_times, w, label="A*", color="#0094d4")
    ax1.bar(x + w / 2, r_times, w, label="RRT*", color="#7c3aed", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax1.set_ylabel("Time (ms)")

    title = (f"A* vs RRT* by room  |  "
             f"parallel({n_workers}w): {par_time:.2f}s")
    if seq_time is not None:
        title += f"  vs  sequential: {seq_time:.2f}s  ({seq_time / par_time:.2f}x speedup)"
    ax1.set_title(title, fontsize=11, fontweight="bold")
    ax1.legend();
    ax1.grid(axis="y", alpha=0.3)

    a_dists = [r["astar_distance"] for r in valid]
    r_dists = [r["rrt_distance"] if r["rrt_success"] else 0 for r in valid]
    ax2.bar(x - w / 2, a_dists, w, label="A*", color="#0094d4")
    ax2.bar(x + w / 2, r_dists, w, label="RRT*", color="#7c3aed", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax2.set_ylabel("Path length (m)")
    ax2.set_title("Path length comparison by room")
    ax2.legend();
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"\n그래프 저장: {output.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy-dir", type=Path, default=Path("npy"))
    parser.add_argument("--rooms-json", type=Path, default=Path("rooms_graph.json"),
                        help="rooms_graph.json 경로 (통로 정보 활용)")  # ✨ [추가]
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--resolution", type=float, default=0.15)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--rrt-iter", type=int, default=3000)
    parser.add_argument("--rrt-step", type=float, default=0.50)
    parser.add_argument("--rrt-radius", type=float, default=1.50)
    parser.add_argument("--rrt-bias", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--global-start", type=float, nargs=3,
                        default=[-1.5, -4.0, 0.5])
    parser.add_argument("--global-goal", type=float, nargs=3,
                        default=[14.0, 6.5, 4.0])
    parser.add_argument("--output", type=Path,
                        default=Path("parallel_rooms_result.png"))
    parser.add_argument("--skip-sequential", action="store_true",
                        help="순차 실행 측정 건너뛰기 (시간 절약)")
    args = parser.parse_args()

    # ✨ [추가] rooms_graph.json 로드
    rooms_meta, npy_to_room = load_rooms_metadata(args.rooms_json)
    if rooms_meta:
        print(f"✓ rooms_graph.json 로드: {len(rooms_meta)}개 방")
    else:
        print(f"⚠ rooms_graph.json 로드 실패, 기본값으로 진행")

    folders = sorted(glob.glob(str(args.npy_dir / "*")))
    folders = [f for f in folders if os.path.isdir(f)]
    if not folders:
        print(f"[에러] 폴더 없음: {args.npy_dir}");
        return

    print("=" * 70)
    print("  병렬 vs 순차 A*/RRT* 벤치마크  (rooms_graph.json 기반)")  # ✨ [수정]
    print("=" * 70)
    print(f"  방 개수:          {len(folders)}")
    print(f"  병렬 워커:        {args.workers}")
    print(f"  Resolution:       {args.resolution}m")
    print(f"  전역 시작:        {tuple(args.global_start)}")
    print(f"  전역 목표:        {tuple(args.global_goal)}")
    print(f"  순차 측정:        {'OFF (건너뜀)' if args.skip_sequential else 'ON'}")
    print(f"  Metadata:         {'JSON 로드됨' if rooms_meta else '기본값 사용'}")  # ✨ [추가]
    print("=" * 70 + "\n")

    # ✨ [수정] rooms_meta, npy_to_room 추가
    job_args = [
        (folder, args.resolution, args.sample, args.margin,
         args.rrt_iter, args.rrt_step, args.rrt_radius, args.rrt_bias,
         args.seed, args.global_start, args.global_goal,
         rooms_meta, npy_to_room)
        for folder in folders
    ]

    # ── 병렬 실행 ────────────────────────────────────────────────────
    print(f"[1/2] 병렬 실행 시작 ({args.workers} 워커)...\n")
    t_par = time.perf_counter()
    with Pool(processes=args.workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(process_room, job_args), 1):
            short = r["room"].replace("00809_Qpor2mEya8F_", "")
            print(f"  [{i:2d}/{len(folders)}] 완료: {short}")
            results.append(r)
    par_time = time.perf_counter() - t_par
    print(f"\n  → 병렬 실행 완료: {par_time:.2f}초\n")

    results.sort(key=lambda r: r["room"])

    # ── 순차 실행 (옵션) ────────────────────────────────────────
    seq_time = None
    if not args.skip_sequential:
        print(f"[2/2] 순차 실행 시작 (for loop, 1 프로세스)...")
        print(f"      ⏳ 시간 소요 예상: 약 {par_time * args.workers:.0f}초 정도\n")
        t_seq = time.perf_counter()
        for i, ja in enumerate(job_args, 1):
            r = process_room(ja)
            short = r["room"].replace("00809_Qpor2mEya8F_", "")
            print(f"  [{i:2d}/{len(folders)}] 완료: {short}")
        seq_time = time.perf_counter() - t_seq
        print(f"\n  → 순차 실행 완료: {seq_time:.2f}초\n")
    else:
        print("[2/2] 순차 실행 건너뜀 (--skip-sequential)\n")

    # ── 결과 ─────────────────────────────────────────────────────
    print_table(results, par_time, args.workers, seq_time)
    plot_results(results, args.output, par_time, args.workers, seq_time)


if __name__ == "__main__":
    main()