#!/usr/bin/env python3
"""
Compare A* and RRT* on the same 3D start/goal pair and draw matplotlib charts.

Examples:
  python compare_rrtstar_astar.py
  python compare_rrtstar_astar.py --input scene.npy
  python compare_rrtstar_astar.py --input scene.ply --runs 20 --show
"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("/Users/yoochaewon/Desktop/Qpor2mEya8F_voxel.npy")


@dataclass
class GridMeta:
    grid: np.ndarray
    x_min: float
    y_min: float
    z_min: float
    resolution: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    @property
    def free_ratio(self) -> float:
        return float(np.mean(self.grid == 0))


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 3)
        if arr.shape[1] < 3:
            raise ValueError("NPY must contain at least 3 columns: x, y, z")
        return arr[:, :3]

    if path.suffix.lower() == ".ply":
        return load_ply_points(path)

    raise ValueError("Only .npy and .ply point clouds are supported by this script.")


def load_ply_points(path: Path) -> np.ndarray:
    data = path.read_bytes()
    marker = b"end_header"
    header_end = data.find(marker)
    if header_end < 0:
        raise ValueError("PLY header does not contain end_header")
    header_end += len(marker)
    if data[header_end : header_end + 2] == b"\r\n":
        header_end += 2
    elif data[header_end : header_end + 1] == b"\n":
        header_end += 1

    header = data[:header_end].decode("utf-8", errors="replace")
    vertex_count = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
            in_vertex = True
            continue
        if len(parts) >= 2 and parts[0] == "element" and parts[1] != "vertex":
            in_vertex = False
        if in_vertex and len(parts) >= 3 and parts[0] == "property":
            properties.append((parts[1], parts[2]))

    if vertex_count is None:
        raise ValueError("PLY vertex count not found")

    names = [name for _, name in properties]
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY must contain x, y, z vertex properties")

    if "format ascii" in header:
        rows = []
        text = data[header_end:].decode("utf-8", errors="replace").splitlines()
        x_i, y_i, z_i = names.index("x"), names.index("y"), names.index("z")
        for line in text[:vertex_count]:
            vals = line.split()
            rows.append((float(vals[x_i]), float(vals[y_i]), float(vals[z_i])))
        return np.asarray(rows, dtype=float)

    if "format binary_little_endian" not in header:
        raise ValueError("Only ascii and binary_little_endian PLY are supported.")

    sizes = {
        "char": 1,
        "uchar": 1,
        "int8": 1,
        "uint8": 1,
        "short": 2,
        "ushort": 2,
        "int16": 2,
        "uint16": 2,
        "int": 4,
        "uint": 4,
        "int32": 4,
        "uint32": 4,
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
    }
    formats = {
        "char": "b",
        "uchar": "B",
        "int8": "b",
        "uint8": "B",
        "short": "h",
        "ushort": "H",
        "int16": "h",
        "uint16": "H",
        "int": "i",
        "uint": "I",
        "int32": "i",
        "uint32": "I",
        "float": "f",
        "float32": "f",
        "double": "d",
        "float64": "d",
    }
    stride = sum(sizes[t] for t, _ in properties)
    offsets = {}
    offset = 0
    for typ, name in properties:
        offsets[name] = (offset, formats[typ])
        offset += sizes[typ]

    points = np.empty((vertex_count, 3), dtype=float)
    for i in range(vertex_count):
        base = header_end + i * stride
        for col, name in enumerate(("x", "y", "z")):
            off, fmt = offsets[name]
            points[i, col] = struct.unpack_from("<" + fmt, data, base + off)[0]
    return points


def make_synthetic_points() -> np.ndarray:
    xs = np.arange(-4.0, 4.2, 0.2)
    ys = np.arange(-2.0, 2.2, 0.2)
    zs = np.arange(0.0, 3.2, 0.2)
    pts = []

    def add_box(x0, x1, y0, y1, z0, z1):
        for x in xs:
            for y in ys:
                for z in zs:
                    on_surface = (
                        abs(x - x0) < 1e-9
                        or abs(x - x1) < 1e-9
                        or abs(y - y0) < 1e-9
                        or abs(y - y1) < 1e-9
                        or abs(z - z0) < 1e-9
                        or abs(z - z1) < 1e-9
                    )
                    if x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1 and on_surface:
                        pts.append((x, y, z))

    add_box(-0.6, 0.6, -1.2, 1.2, 0.0, 2.4)
    add_box(1.6, 2.2, -1.8, -0.2, 0.0, 2.2)
    add_box(-2.2, -1.4, 0.3, 1.8, 0.0, 2.7)
    return np.asarray(pts, dtype=float)


def voxelize(points: np.ndarray, resolution: float, margin: int, sample: int) -> GridMeta:
    sampled = points[:: max(1, sample)]
    xyz_min = sampled.min(axis=0)
    xyz_max = sampled.max(axis=0)
    shape = np.ceil((xyz_max - xyz_min) / resolution).astype(int) + 3
    grid = np.zeros(tuple(shape), dtype=np.uint8)

    ijk = np.rint((sampled - xyz_min) / resolution).astype(int)
    valid = np.all((ijk >= 0) & (ijk < shape), axis=1)
    grid[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]] = 1

    if margin > 0:
        inflated = grid.copy()
        occupied = np.argwhere(grid == 1)
        for ix, iy, iz in occupied:
            x0, x1 = max(0, ix - margin), min(shape[0], ix + margin + 1)
            y0, y1 = max(0, iy - margin), min(shape[1], iy + margin + 1)
            z0, z1 = max(0, iz - margin), min(shape[2], iz + margin + 1)
            inflated[x0:x1, y0:y1, z0:z1] = 1
        grid = inflated

    return GridMeta(grid, xyz_min[0], xyz_min[1], xyz_min[2], resolution)


def world_to_voxel(gm: GridMeta, p: np.ndarray | tuple[float, float, float]) -> tuple[int, int, int]:
    x, y, z = p
    return (
        round((x - gm.x_min) / gm.resolution),
        round((y - gm.y_min) / gm.resolution),
        round((z - gm.z_min) / gm.resolution),
    )


def voxel_to_world(gm: GridMeta, v: tuple[int, int, int]) -> np.ndarray:
    ix, iy, iz = v
    return np.array(
        [gm.x_min + ix * gm.resolution, gm.y_min + iy * gm.resolution, gm.z_min + iz * gm.resolution],
        dtype=float,
    )


def grid_at(gm: GridMeta, ix: int, iy: int, iz: int) -> int:
    nx, ny, nz = gm.shape
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return 1
    return int(gm.grid[ix, iy, iz])


def find_nearest_free(gm: GridMeta, v: tuple[int, int, int], max_radius: int = 20):
    ix, iy, iz = v
    if grid_at(gm, ix, iy, iz) == 0:
        return v
    for radius in range(1, max_radius + 1):
        best = None
        best_d = math.inf
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius and abs(dz) != radius:
                        continue
                    candidate = (ix + dx, iy + dy, iz + dz)
                    if grid_at(gm, *candidate) == 0:
                        d = dx * dx + dy * dy + dz * dz
                        if d < best_d:
                            best = candidate
                            best_d = d
        if best is not None:
            return best
    return None


def line_of_sight(gm: GridMeta, a: np.ndarray, b: np.ndarray, steps: int | None = None) -> bool:
    dist = float(np.linalg.norm(b - a))
    n = steps or max(2, math.ceil(dist / (gm.resolution * 0.7)))
    for t in np.linspace(0.0, 1.0, n + 1):
        p = a + (b - a) * t
        if grid_at(gm, *world_to_voxel(gm, p)) != 0:
            return False
    return True


def smooth_path(path: list[np.ndarray], gm: GridMeta) -> list[np.ndarray]:
    if len(path) < 3:
        return path
    result = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if line_of_sight(gm, path[i], path[j], steps=30):
                break
            j -= 1
        i = j
        result.append(path[i])
    return result


def astar(gm: GridMeta, start: np.ndarray, goal: np.ndarray, max_iters: int = 200_000):
    sv = find_nearest_free(gm, world_to_voxel(gm, start))
    ev = find_nearest_free(gm, world_to_voxel(gm, goal))
    if sv is None or ev is None:
        return None, {"expanded": 0}

    nx, ny, nz = gm.shape

    def encode(v):
        return v[0] * ny * nz + v[1] * nz + v[2]

    def decode(k):
        ix = k // (ny * nz)
        iy = (k % (ny * nz)) // nz
        iz = k % nz
        return int(ix), int(iy), int(iz)

    def heuristic(v):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v, ev)))

    dirs = [
        (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == dy == dz == 0)
    ]

    start_key = encode(sv)
    goal_key = encode(ev)
    came_from = {start_key: -1}
    g_score = {start_key: 0.0}
    heap = [(heuristic(sv), 0.0, sv)]
    expanded = 0

    while heap:
        _, g, cur = heapq.heappop(heap)
        key = encode(cur)
        if g > g_score.get(key, math.inf) + 1e-9:
            continue

        expanded += 1
        if expanded > max_iters:
            return None, {"expanded": expanded}

        if key == goal_key:
            path = []
            while key != -1:
                path.append(voxel_to_world(gm, decode(key)))
                key = came_from[key]
            path.reverse()
            return path, {"expanded": expanded}

        ix, iy, iz = cur
        for dx, dy, dz, step in dirs:
            nxt = (ix + dx, iy + dy, iz + dz)
            if grid_at(gm, *nxt) != 0:
                continue
            nk = encode(nxt)
            ng = g + step
            if ng < g_score.get(nk, math.inf):
                g_score[nk] = ng
                came_from[nk] = key
                heapq.heappush(heap, (ng + heuristic(nxt), ng, nxt))

    return None, {"expanded": expanded}


def rrt_star(
    gm: GridMeta,
    start_world: np.ndarray,
    goal_world: np.ndarray,
    rng: np.random.Generator,
    max_iter: int,
    step_len: float,
    rewire_radius: float,
    goal_bias: float,
):
    sv = find_nearest_free(gm, world_to_voxel(gm, start_world))
    ev = find_nearest_free(gm, world_to_voxel(gm, goal_world))
    if sv is None or ev is None:
        return None, {"nodes": 0}

    start = voxel_to_world(gm, sv)
    goal = voxel_to_world(gm, ev)
    shape = np.array(gm.shape, dtype=float)
    lo = np.array([gm.x_min, gm.y_min, gm.z_min]) - 1.0
    hi = np.array([gm.x_min, gm.y_min, gm.z_min]) + shape * gm.resolution + 1.0

    nodes = [{"p": start, "parent": -1, "cost": 0.0}]
    best_goal_idx = -1
    best_goal_cost = math.inf

    def nearest(q):
        dists = [float(np.linalg.norm(n["p"] - q)) for n in nodes]
        return int(np.argmin(dists))

    def near(q):
        return [i for i, n in enumerate(nodes) if float(np.linalg.norm(n["p"] - q)) <= rewire_radius]

    for _ in range(max_iter):
        if rng.random() < goal_bias:
            q = goal
        else:
            q = rng.uniform(lo, hi)

        ni = nearest(q)
        base = nodes[ni]
        d = float(np.linalg.norm(q - base["p"]))
        if d == 0:
            continue
        new_p = base["p"] + (q - base["p"]) * min(step_len / d, 1.0)

        if not line_of_sight(gm, base["p"], new_p):
            continue

        near_ids = near(new_p)
        parent = ni
        best_cost = base["cost"] + float(np.linalg.norm(new_p - base["p"]))
        for idx in near_ids:
            candidate = nodes[idx]
            cost = candidate["cost"] + float(np.linalg.norm(new_p - candidate["p"]))
            if cost < best_cost and line_of_sight(gm, candidate["p"], new_p):
                parent = idx
                best_cost = cost

        new_idx = len(nodes)
        nodes.append({"p": new_p, "parent": parent, "cost": best_cost})

        for idx in near_ids:
            if idx == parent:
                continue
            candidate = nodes[idx]
            new_cost = best_cost + float(np.linalg.norm(candidate["p"] - new_p))
            if new_cost < candidate["cost"] and line_of_sight(gm, new_p, candidate["p"]):
                candidate["parent"] = new_idx
                candidate["cost"] = new_cost

        d_goal = float(np.linalg.norm(goal - new_p))
        if d_goal < step_len and line_of_sight(gm, new_p, goal):
            goal_cost = best_cost + d_goal
            if goal_cost < best_goal_cost:
                best_goal_cost = goal_cost
                best_goal_idx = len(nodes)
                nodes.append({"p": goal, "parent": new_idx, "cost": goal_cost})

    if best_goal_idx == -1:
        return None, {"nodes": len(nodes)}

    path = []
    cur = best_goal_idx
    while cur != -1:
        path.append(nodes[cur]["p"])
        cur = nodes[cur]["parent"]
    path.reverse()
    return path, {"nodes": len(nodes)}


def path_length(path: list[np.ndarray] | None) -> float:
    if not path or len(path) < 2:
        return math.inf
    return float(sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))))


def run_once(name: str, func, *args, smooth=False, gm=None, **kwargs):
    t0 = time.perf_counter()
    path, info = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if smooth and path is not None:
        path = smooth_path(path, gm)
    return {
        "algorithm": name,
        "success": path is not None,
        "time_ms": elapsed_ms,
        "distance_m": path_length(path),
        "waypoints": len(path) if path is not None else 0,
        **info,
    }


def plot_results(results: list[dict], output: Path, show: bool) -> None:
    labels = ["A*", "RRT*"]
    metrics = [
        ("time_ms", "Time (ms)", "lower is better"),
        ("distance_m", "Path length (m)", "lower is better"),
        ("waypoints", "Waypoints", "lower is simpler"),
        ("success", "Success rate", "higher is better"),
    ]
    colors = {"A*": "#0094d4", "RRT*": "#7c3aed"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    for ax, (key, title, subtitle) in zip(axes, metrics):
        values = []
        err = []
        for label in labels:
            vals = [r[key] for r in results if r["algorithm"] == label]
            if key == "success":
                vals = [100.0 if v else 0.0 for v in vals]
            finite = [v for v in vals if np.isfinite(v)]
            values.append(float(np.mean(finite)) if finite else 0.0)
            err.append(float(np.std(finite)) if finite else 0.0)

        ax.bar(labels, values, yerr=err if key != "success" else None, color=[colors[x] for x in labels], alpha=0.9)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        if key == "success":
            ax.set_ylim(0, 105)
            ax.set_ylabel("%")

    fig.suptitle("A* vs RRT* Efficiency on Same Start/Goal", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def print_summary(results: list[dict]) -> None:
    print("\n=== Summary ===")
    for algo in ("A*", "RRT*"):
        rows = [r for r in results if r["algorithm"] == algo]
        success = [r for r in rows if r["success"]]
        rate = len(success) / len(rows) * 100.0
        print(f"\n{algo}")
        print(f"  success: {rate:.1f}% ({len(success)}/{len(rows)})")
        if success:
            print(f"  avg time: {np.mean([r['time_ms'] for r in success]):.2f} ms")
            print(f"  avg distance: {np.mean([r['distance_m'] for r in success]):.2f} m")
            print(f"  avg waypoints: {np.mean([r['waypoints'] for r in success]):.1f}")
            if algo == "A*":
                print(f"  avg expanded voxels: {np.mean([r['expanded'] for r in success]):.1f}")
            else:
                print(f"  avg nodes: {np.mean([r['nodes'] for r in success]):.1f}")

    a = [r for r in results if r["algorithm"] == "A*" and r["success"]]
    r = [r for r in results if r["algorithm"] == "RRT*" and r["success"]]
    if a and r:
        a_t = np.mean([x["time_ms"] for x in a])
        r_t = np.mean([x["time_ms"] for x in r])
        a_d = np.mean([x["distance_m"] for x in a])
        r_d = np.mean([x["distance_m"] for x in r])
        print("\nRelative")
        print(f"  RRT* time / A* time: {r_t / a_t:.2f}x")
        print(f"  RRT* distance change vs A*: {(r_d - a_d) / a_d * 100.0:+.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Optional .npy or .ply point cloud")
    parser.add_argument("--runs", type=int, default=10, help="RRT* repeats; A* is repeated too for comparable timing")
    parser.add_argument("--resolution", type=float, default=0.15)
    parser.add_argument("--margin", type=int, default=0, help="Obstacle inflation in voxels; this voxel NPY works best with 0")
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--start", type=float, nargs=3, default=(-3.0, -0.5, 1.5))
    parser.add_argument("--goal", type=float, nargs=3, default=(3.5, 0.5, 2.0))
    parser.add_argument("--rrt-iter", type=int, default=3000)
    parser.add_argument("--rrt-step", type=float, default=0.50)
    parser.add_argument("--rrt-radius", type=float, default=1.50)
    parser.add_argument("--rrt-bias", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("algorithm_comparison.png"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    points = load_points(args.input) if args.input else make_synthetic_points()
    gm = voxelize(points, args.resolution, args.margin, args.sample)
    start = np.asarray(args.start, dtype=float)
    goal = np.asarray(args.goal, dtype=float)

    print(f"Grid shape: {gm.shape}, free space: {gm.free_ratio * 100:.1f}%")
    print(f"Start: {tuple(start)}, Goal: {tuple(goal)}")

    rng = np.random.default_rng(args.seed)
    results = []
    for _ in range(args.runs):
        results.append(run_once("A*", astar, gm, start, goal, smooth=True, gm=gm))
        results.append(
            run_once(
                "RRT*",
                rrt_star,
                gm,
                start,
                goal,
                rng,
                args.rrt_iter,
                args.rrt_step,
                args.rrt_radius,
                args.rrt_bias,
            )
        )

    print_summary(results)
    plot_results(results, args.output, args.show)
    print(f"\nSaved graph: {args.output.resolve()}")


if __name__ == "__main__":
    main()
