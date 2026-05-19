"""
main.py
=======
CLI entry point for multi-room parallel A* path planning.

Usage examples
--------------
# Basic run with synthetic data (no --data-dir required)
python main.py --start sub002 --goal sub011

# With real NPY data
python main.py --data-dir /path/to/data --start sub002 --goal sub011 --runs 5

# Customise resolution, workers, output folder
python main.py --start sub002 --goal sub005 \\
               --resolution 0.10 --margin 1 \\
               --workers 4 --runs 8 --output-dir results/ --show
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from parallel import config as cfg
from benchmark import plot_benchmark, print_summary, run_benchmark
from graph import bfs_path, build_graph, describe_path, dfs_path
from parallel_planner import (
    build_tasks,
    run_adaptive,
    stitch_paths,
)
from voxel_io import GridMeta, load_all_rooms


# ---------------------------------------------------------------------------
# Synthetic scene generator (demo / CI mode)
# ---------------------------------------------------------------------------

def make_synthetic_grid(room_id: str, seed: int = 0) -> GridMeta:
    """
    Return a small procedural voxel grid so the pipeline can run without real
    NPY files.  Obstacles are randomly placed boxes.
    """
    rng = np.random.default_rng(seed)
    SIZE = 40
    grid = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)

    # Floor & ceiling
    grid[:, :, 0] = 1
    grid[:, :, SIZE - 1] = 1

    # Random box obstacles
    n_boxes = rng.integers(3, 8)
    for _ in range(n_boxes):
        x0, y0, z0 = rng.integers(5, SIZE - 15, size=3)
        dx, dy, dz = rng.integers(3, 10, size=3)
        grid[x0:x0+dx, y0:y0+dy, z0:z0+dz] = 1

    return GridMeta(
        grid=grid,
        x_min=float(rng.uniform(-6, -4)),
        y_min=float(rng.uniform(0, 2)),
        z_min=0.0,
        resolution=0.12,
        room_id=room_id,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-room parallel A* path planner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Input ---
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Root folder with one sub-directory per room. "
                        "Omit to run on synthetic grids (demo mode).")
    p.add_argument("--start", default="sub002",
                   help="Source room ID, e.g. sub002")
    p.add_argument("--goal",  default="sub011",
                   help="Destination room ID, e.g. sub011")

    # --- World coordinates ---
    p.add_argument("--start-xyz", type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Start position in world space.  "
                        "Defaults to the centroid of the start room.")
    p.add_argument("--goal-xyz",  type=float, nargs=3, default=None,
                   metavar=("X", "Y", "Z"),
                   help="Goal position in world space.  "
                        "Defaults to the centroid of the goal room.")

    # --- Voxelisation ---
    p.add_argument("--resolution", type=float, default=cfg.RESOLUTION)
    p.add_argument("--margin",     type=int,   default=cfg.MARGIN)
    p.add_argument("--sample",     type=int,   default=cfg.POINT_SAMPLE_STEP)

    # --- A* ---
    p.add_argument("--max-iter", type=int, default=cfg.ASTAR_MAX_ITER)

    # --- Graph search ---
    p.add_argument("--graph-algo", choices=["bfs", "dfs"], default="bfs",
                   help="Algorithm for room-sequence search.")

    # --- Parallel ---
    p.add_argument("--workers", type=int, default=None,
                   help="Worker processes (default: os.cpu_count())")

    # --- Benchmark ---
    p.add_argument("--runs",   type=int,  default=cfg.BENCHMARK_RUNS)
    p.add_argument("--output-dir", type=Path, default=Path("../results"))
    p.add_argument("--show",   action="store_true",
                   help="Open chart window (requires display).")
    p.add_argument("--no-benchmark", action="store_true",
                   help="Skip benchmark; just print the path.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Build room graph and find room sequence
    # ------------------------------------------------------------------
    graph = build_graph(cfg.ROOM_GRAPH)
    search_fn = bfs_path if args.graph_algo == "bfs" else dfs_path

    print(f"\n[Graph] Finding room sequence: {args.start} → {args.goal}  "
          f"({args.graph_algo.upper()})")
    room_sequence = search_fn(graph, args.start, args.goal)
    if room_sequence is None:
        print(f"[Error] No path found between {args.start!r} and {args.goal!r}.",
              file=sys.stderr)
        return 1
    print(f"[Graph] {describe_path(room_sequence)}")

    # ------------------------------------------------------------------
    # 2. Load / generate voxel grids
    # ------------------------------------------------------------------
    print("\n[Voxels] Loading room grids …")
    if args.data_dir is not None:
        grids = load_all_rooms(
            args.data_dir,
            room_ids=room_sequence,
            resolution=args.resolution,
            margin=args.margin,
            sample_step=args.sample,
        )
    else:
        print("  (no --data-dir given; using synthetic grids for demo)")
        grids = {
            rid: make_synthetic_grid(rid, seed=int(rid.replace("sub", "")))
            for rid in room_sequence
        }

    for rid, gm in grids.items():
        print(f"  {rid}: {gm}")

    # ------------------------------------------------------------------
    # 3. Resolve world-space start / goal
    # ------------------------------------------------------------------
    def centroid(gm: GridMeta) -> np.ndarray:
        return gm.origin + np.array(gm.shape) * gm.resolution / 2.0

    start_world = (
        np.array(args.start_xyz, dtype=float)
        if args.start_xyz is not None
        else centroid(grids[room_sequence[0]])
    )
    goal_world = (
        np.array(args.goal_xyz, dtype=float)
        if args.goal_xyz is not None
        else centroid(grids[room_sequence[-1]])
    )

    print(f"\n[Plan]  Start : {start_world.round(3).tolist()}")
    print(f"[Plan]  Goal  : {goal_world.round(3).tolist()}")

    # ------------------------------------------------------------------
    # 4. Build task list
    # ------------------------------------------------------------------
    tasks = build_tasks(
        room_sequence=room_sequence,
        grids=grids,
        start_world=start_world,
        goal_world=goal_world,
        max_iter=args.max_iter,
    )
    print(f"\n[Tasks] {len(tasks)} segments created.")

    # ------------------------------------------------------------------
    # 5. Quick single parallel run (always shown)
    # ------------------------------------------------------------------
    print("\n[Run] Executing adaptive planner …")

    par_results, par_wall, mode = run_adaptive(
        tasks,
        n_workers=args.workers,
    )

    print(f"[Run] Selected mode: {mode}")
    full_path = stitch_paths(par_results, room_sequence)

    if full_path is None:
        print("[Result] ✗ Planning failed on one or more segments.",
              file=sys.stderr)
        for r in par_results:
            status = "✓" if r["success"] else "✗"
            print(f"  {status} {r['room_id']:10s}  "
                  f"{r['time_ms']:7.1f} ms  "
                  f"expanded={r['expanded']:,}")
        return 1

    from astar import path_length
    total_len = path_length(full_path)
    print(f"[Result] ✓ Path found — {len(full_path)} waypoints, "
          f"{total_len:.2f} m total, {par_wall:.1f} ms wall time")

    # ------------------------------------------------------------------
    # 6. Benchmark (optional)
    # ------------------------------------------------------------------
    if not args.no_benchmark:
        print(f"\n[Benchmark] {args.runs} runs × (sequential + parallel) …")
        data = run_benchmark(
            tasks=tasks,
            room_sequence=room_sequence,
            n_runs=args.runs,
            n_workers=args.workers,
        )
        print_summary(data)
        ts = time.strftime("%Y%m%d_%H%M%S")
        plot_benchmark(data, output_dir=args.output_dir, show=args.show, timestamp=ts)

    return 0


if __name__ == "__main__":
    sys.exit(main())
