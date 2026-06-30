#!/usr/bin/env python3
"""
방별 npy 파일을 병렬로 A*/RRT* 처리하는 벤치마크.
+ 실제 순차 실행 시간도 측정해서 진짜 가속비 비교.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
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

voxelize    = pathfinder.voxelize
astar       = pathfinder.astar
rrt_star    = pathfinder.rrt_star
smooth_path = pathfinder.smooth_path
path_length = pathfinder.path_length


# ──────────────────────────────────────────────────────────────────────────────
# 한 방 처리
# ──────────────────────────────────────────────────────────────────────────────
def process_room(args):
    (folder_path, resolution, sample, margin,
     rrt_iter, rrt_step, rrt_radius, rrt_bias, seed,
     global_start, global_goal) = args

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
        contains_goal  = bool(np.all(xyz_min <= gg) and np.all(gg <= xyz_max))

        delta = (xyz_max - xyz_min) * 0.1
        start = xyz_min + delta
        goal  = xyz_max - delta

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
            "n_points": len(points),
            "diagonal_m": float(np.linalg.norm(goal - start)),
            "contains_global_start": contains_start,
            "contains_global_goal": contains_goal,

            "astar_success":   a_path is not None,
            "astar_time_ms":   a_time,
            "astar_distance":  path_length(a_path) if a_path else None,
            "astar_waypoints": len(a_path) if a_path else 0,

            "rrt_success":   r_path is not None,
            "rrt_time_ms":   r_time,
            "rrt_distance":  path_length(r_path) if r_path else None,
            "rrt_waypoints": len(r_path) if r_path else 0,

            "error": None,
        }
    except Exception as e:
        return {"room": name, "error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# 결과 출력
# ──────────────────────────────────────────────────────────────────────────────
def print_table(results, par_time, n_workers, seq_time=None):
    print("\n" + "=" * 115)
    print(f"{'Room':<20} {'대각선':>8} {'A* 시간':>11} {'A* 거리':>9} "
          f"{'RRT* 시간':>13} {'RRT* 거리':>11} {'전역점':>8}")
    print("-" * 115)

    a_times, r_times = [], []
    start_room, goal_room = None, None

    for r in results:
        if r.get("error"):
            short = r["room"].replace("00809_Qpor2mEya8F_", "")
            print(f"{short:<20}  [에러] {r['error']}")
            continue

        short = r["room"].replace("00809_Qpor2mEya8F_", "")
        diag = r["diagonal_m"]
        a_t = f"{r['astar_time_ms']:>10.1f}" if r["astar_success"] else "      실패"
        a_d = f"{r['astar_distance']:>8.2f}" if r["astar_success"] else "       -"
        r_t = f"{r['rrt_time_ms']:>12.1f}" if r["rrt_success"] else "        실패"
        r_d = f"{r['rrt_distance']:>10.2f}" if r["rrt_success"] else "         -"

        global_marker = ""
        if r["contains_global_start"]:
            global_marker = "🟢시작"
            start_room = short
        if r["contains_global_goal"]:
            global_marker = "🔴목표"
            goal_room = short

        print(f"{short:<20} {diag:>6.2f}m {a_t}ms {a_d}m {r_t}ms {r_d}m {global_marker:>8}")

        if r["astar_success"]: a_times.append(r["astar_time_ms"])
        if r["rrt_success"]:   r_times.append(r["rrt_time_ms"])

    print("=" * 115)

    print(f"\n[전역 좌표 분석]")
    print(f"  시작 좌표가 속한 방:  {start_room or '(어느 방에도 안 속함)'}")
    print(f"  목표 좌표가 속한 방:  {goal_room or '(어느 방에도 안 속함)'}")

    if a_times:
        seq_calc = sum(a_times) + sum(r_times)
        print(f"\n[A*]   성공 {len(a_times)}개, 평균 {np.mean(a_times):.1f}ms, "
              f"최대 {max(a_times):.1f}ms")
        print(f"[RRT*] 성공 {len(r_times)}개, 평균 {np.mean(r_times):.1f}ms, "
              f"최대 {max(r_times):.1f}ms")

        print(f"\n{'─' * 65}")
        print(f"  ① 환산 순차 시간  (방 시간 단순 합):      {seq_calc/1000:>7.2f}초")
        if seq_time is not None:
            print(f"  ② 실제 순차 시간  (for loop 실측):        {seq_time:>7.2f}초")
        print(f"  ③ 실제 병렬 시간  ({n_workers} 워커, 실측):       {par_time:>7.2f}초")
        print(f"{'─' * 65}")
        print(f"  환산 기준 가속비  (① / ③):                {seq_calc/1000/par_time:>7.2f}x")
        if seq_time is not None:
            print(f"  실측 기준 가속비  (② / ③):                {seq_time/par_time:>7.2f}x  ⭐")
        print(f"{'─' * 65}")


def plot_results(results, output, par_time, n_workers, seq_time=None):
    valid = [r for r in results if not r.get("error") and r["astar_success"]]
    if not valid:
        print("그래프 그릴 데이터 없음"); return

    names   = [r["room"].replace("00809_Qpor2mEya8F_", "") for r in valid]
    a_times = [r["astar_time_ms"] for r in valid]
    r_times = [r["rrt_time_ms"] if r["rrt_success"] else 0 for r in valid]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
    x = np.arange(len(names))
    w = 0.4

    ax1.bar(x - w/2, a_times, w, label="A*",   color="#0094d4")
    ax1.bar(x + w/2, r_times, w, label="RRT*", color="#7c3aed", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax1.set_ylabel("Time (ms)")

    title = (f"A* vs RRT* by room  |  "
             f"parallel({n_workers}w): {par_time:.2f}s")
    if seq_time is not None:
        title += f"  vs  sequential: {seq_time:.2f}s  ({seq_time/par_time:.2f}x speedup)"
    ax1.set_title(title, fontsize=11, fontweight="bold")
    ax1.legend(); ax1.grid(axis="y", alpha=0.3)

    a_dists = [r["astar_distance"] for r in valid]
    r_dists = [r["rrt_distance"] if r["rrt_success"] else 0 for r in valid]
    ax2.bar(x - w/2, a_dists, w, label="A*",   color="#0094d4")
    ax2.bar(x + w/2, r_dists, w, label="RRT*", color="#7c3aed", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax2.set_ylabel("Path length (m)")
    ax2.set_title("Path length comparison by room")
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)

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
    parser.add_argument("--global-goal",  type=float, nargs=3,
                        default=[14.0, 6.5, 4.0])
    parser.add_argument("--output", type=Path,
                        default=Path("parallel_rooms_result.png"))
    parser.add_argument("--skip-sequential", action="store_true",
                        help="순차 실행 측정 건너뛰기 (시간 절약)")
    args = parser.parse_args()

    folders = sorted(glob.glob(str(args.npy_dir / "*")))
    folders = [f for f in folders if os.path.isdir(f)]
    if not folders:
        print(f"[에러] 폴더 없음: {args.npy_dir}"); return

    print("=" * 60)
    print("  병렬 vs 순차 A*/RRT* 벤치마크")
    print("=" * 60)
    print(f"  방 개수:      {len(folders)}")
    print(f"  병렬 워커:    {args.workers}")
    print(f"  Resolution:   {args.resolution}m")
    print(f"  전역 시작:    {tuple(args.global_start)}")
    print(f"  전역 목표:    {tuple(args.global_goal)}")
    print(f"  순차 측정:    {'OFF (건너뜀)' if args.skip_sequential else 'ON'}")
    print("=" * 60 + "\n")

    job_args = [
        (folder, args.resolution, args.sample, args.margin,
         args.rrt_iter, args.rrt_step, args.rrt_radius, args.rrt_bias,
         args.seed, args.global_start, args.global_goal)
        for folder in folders
    ]

    # ── 병렬 실행 ────────────────────────────────────────────────
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