# Multi-Room 3D Pathfinding with Parallel A*

> Parallel A\* path planning across multiple rooms of a 3D indoor scene  
> (HM3D scene **TEEsavR23oF**, 10 rooms), with BFS/DFS room-sequence search,  
> `multiprocessing` parallelism, and publication-quality benchmark charts.

---

## Pipeline

```
User Input: start room + goal room
        │
        ▼
  Room Graph (BFS/DFS)  ──► ordered room sequence
        │
        ▼
  build_tasks()   ──► per-room (start_xyz, goal_xyz, GridMeta) tuples
        │                       ↑ door waypoints stitch segments together
        ▼
  multiprocessing.Pool.map(_worker_payload)   ← TRUE parallelism
        │         per-room A* runs concurrently on separate CPU cores
        ▼
  stitch_paths()  ──► single continuous 3-D trajectory
        │
        ▼
  benchmark.py    ──► results/benchmark_<timestamp>.png
```

---

## File Structure

```
multi_room_pathfinding/
├── README.md
├── requirements.txt
├── config.py            # All hyper-parameters + room graph adjacency
├── voxel_io.py          # NPY loading → GridMeta (voxel occupancy grid)
├── graph.py             # Room graph: BFS / DFS / all_paths
├── astar.py             # A* on a single GridMeta; _worker_payload (picklable)
├── parallel_planner.py  # build_tasks, run_sequential, run_parallel, stitch_paths
├── benchmark.py         # Timing, statistics, matplotlib 2×3 panel chart
└── main.py              # CLI entry point
```

---

## Quick Start

```bash
pip install -r requirements.txt

# ── Demo mode (no data needed) ─────────────────────────────────────────────
python main.py --start sub002 --goal sub011 --runs 5

# ── With real NPY data ─────────────────────────────────────────────────────
# Expected layout:
#   data/00800_TEEsavR23oF_000_002/coord.npy
#   data/00800_TEEsavR23oF_000_003/coord.npy
#   ...  (sub002 – sub011, 10 rooms)

python main.py --data-dir data/ --start sub002 --goal sub011 --runs 5
```

---

## Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | *(none)* | Root folder with room sub-directories; omit for synthetic demo |
| `--start` | `sub002` | Source room ID |
| `--goal` | `sub011` | Destination room ID |
| `--start-xyz` | centroid | World-space start position (x y z) |
| `--goal-xyz` | centroid | World-space goal position (x y z) |
| `--resolution` | `0.12` | Voxel grid resolution (metres) |
| `--margin` | `1` | Obstacle inflation radius (voxels) |
| `--sample` | `1` | Down-sample every N-th point cloud point |
| `--max-iter` | `500000` | A* node expansion cap per room |
| `--graph-algo` | `bfs` | `bfs` (shortest hops) or `dfs` |
| `--workers` | CPU count | Parallel worker processes |
| `--runs` | `5` | Benchmark repetitions |
| `--output-dir` | `results/` | PNG output directory |
| `--show` | flag | Open chart window |
| `--no-benchmark` | flag | Skip benchmark; just plan once |

---

## Room Graph (TEEsavR23oF)

Derived from bounding-box analysis of the HM3D GLB scene file.

```
sub002 ── sub004 ── sub003
  │    \       \  /
  │     sub005 ── sub006 ── sub008
  │       │    \       \  /
sub010   sub009  sub011  sub007
  │       │
  └───────┘
```

`config.ROOM_GRAPH` is a plain `dict[str, list[str]]`; update it once door
centroid labels are extracted by the 3-D object-detection branch.

---

## How Parallelism Works

Each room segment is an independent A\* search:

```python
# parallel_planner.py
with Pool(processes=n_workers) as pool:
    results = pool.map(_worker_payload, tasks)
```

`_worker_payload` is a **module-level** function in `astar.py` so that
`multiprocessing` can pickle it on every platform (including Windows with
`spawn` start method).

Door waypoints in `config.DOOR_WAYPOINTS` stitch the segments together:
room _i_ plans from its entry door to its exit door, room _i+1_ starts
exactly where room _i_ ended.

---

## Benchmark Output

`benchmark.py` produces a 2x3 panel PNG:

| Panel | Content |
|-------|---------|
| Wall-clock time | Sequential vs parallel end-to-end latency |
| Sum of worker times | Total CPU work (should be ~equal) |
| Speedup per run | Measured speedup + mean line |
| Per-room breakdown | Time per room x mode (last run) |
| Path length | Total trajectory length |
| Expanded nodes | A\* search effort per room |

On an **8-core machine** with 10 rooms, expect **4-6x** wall-clock speedup.
On a single core the overhead is ~10-20 ms (process spawn); parallelism breaks
even at >= 2 rooms and 2 cores.

---

## Integration Notes (full pipeline)

```bash
python main.py \
  --data-dir data/ \
  --start <room_from_LLM> --goal <room_from_LLM> \
  --start-xyz <x y z>  --goal-xyz <x y z>
```

1. LLM extracts `start_room`, `goal_room`, `target_object` from user speech.
2. 3-D detection maps `target_object` to world-space `goal-xyz`.
3. This script runs the parallel planner and returns the stitched path.
4. Unity consumes the waypoint list for robot/agent navigation.
