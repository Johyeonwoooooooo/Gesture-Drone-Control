#!/usr/bin/env python3
"""
pathway.py — 3D 포인트클라우드 위에서 A*로 시작점→목표점 최적 경로를 찾는 도구.

사용 예:
  python pathway.py --input scene.npy --start -3.0 -0.5 1.5 --goal 3.5 0.5 2.0
  python pathway.py --input scene.ply --start 0 0 0 --goal 5 5 2 --plot
  python pathway.py --input scene.npy --start 0 0 0 --goal 5 5 2 --save path.json

자세한 옵션 설명과 출력 형식은 README.md 참고.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# 격자(voxel grid) 메타데이터
# ──────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────
# 포인트클라우드 로딩 (.npy / .ply)
# ──────────────────────────────────────────────────────────────────────────

def load_points(path: Path) -> np.ndarray:
    """다음 세 가지 입력 형태를 모두 지원한다:
    1) 단일 .npy 또는 .ply 파일
    2) coord.npy가 직접 들어있는 방 폴더 (예: npy/00809_..._room3/)
    3) 여러 방 폴더를 담고 있는 상위 폴더 (예: npy/) → 모든 coord.npy를 합쳐서 하나의 씬으로 사용
    """
    if path.is_dir():
        return _load_points_from_dir(path)

    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _load_npy_file(path)
    if suffix == ".ply":
        return _load_ply_points(path)

    raise ValueError(f"지원하지 않는 입력 형식입니다: {suffix} (.npy, .ply, 또는 폴더만 지원)")


def _load_npy_file(path: Path) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if arr.shape[1] < 3:
        raise ValueError(f"{path}: 최소 3개 컬럼(x, y, z)을 포함해야 합니다.")
    return arr[:, :3]


def _load_points_from_dir(path: Path) -> np.ndarray:
    # 케이스 1: 이 폴더 자체가 하나의 방 폴더 (coord.npy 직접 포함)
    direct_coord = path / "coord.npy"
    if direct_coord.exists():
        print(f"  방 폴더로 인식: {path.name}/coord.npy 로드")
        return _load_npy_file(direct_coord)

    # 케이스 2: 여러 방 폴더를 담은 상위 폴더 → coord.npy를 모두 합치기
    room_dirs = sorted(
        d for d in path.iterdir() if d.is_dir() and (d / "coord.npy").exists()
    )
    if not room_dirs:
        raise ValueError(
            f"{path} 안에서 coord.npy를 찾을 수 없습니다. "
            "이 폴더 또는 하위 폴더에 coord.npy가 있는지 확인하세요."
        )

    print(f"  {len(room_dirs)}개 방 폴더에서 coord.npy 병합 중...")
    all_points = []
    for room_dir in room_dirs:
        pts = _load_npy_file(room_dir / "coord.npy")
        all_points.append(pts)
        print(f"    - {room_dir.name}: {len(pts):,} points")
    return np.concatenate(all_points, axis=0)


def _load_ply_points(path: Path) -> np.ndarray:
    data = path.read_bytes()
    marker = b"end_header"
    header_end = data.find(marker)
    if header_end < 0:
        raise ValueError("PLY 헤더에 end_header가 없습니다.")
    header_end += len(marker)
    if data[header_end:header_end + 2] == b"\r\n":
        header_end += 2
    elif data[header_end:header_end + 1] == b"\n":
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
        raise ValueError("PLY vertex count를 찾지 못했습니다.")

    names = [name for _, name in properties]
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY에 x, y, z 속성이 있어야 합니다.")

    if "format ascii" in header:
        rows = []
        text = data[header_end:].decode("utf-8", errors="replace").splitlines()
        x_i, y_i, z_i = names.index("x"), names.index("y"), names.index("z")
        for line in text[:vertex_count]:
            vals = line.split()
            rows.append((float(vals[x_i]), float(vals[y_i]), float(vals[z_i])))
        return np.asarray(rows, dtype=float)

    if "format binary_little_endian" not in header:
        raise ValueError("ascii / binary_little_endian PLY만 지원합니다.")

    sizes = {
        "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
        "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
        "int": 4, "uint": 4, "int32": 4, "uint32": 4,
        "float": 4, "float32": 4, "double": 8, "float64": 8,
    }
    formats = {
        "char": "b", "uchar": "B", "int8": "b", "uint8": "B",
        "short": "h", "ushort": "H", "int16": "h", "uint16": "H",
        "int": "i", "uint": "I", "int32": "i", "uint32": "I",
        "float": "f", "float32": "f", "double": "d", "float64": "d",
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


# ──────────────────────────────────────────────────────────────────────────
# Voxelize + 격자 유틸
# ──────────────────────────────────────────────────────────────────────────

def voxelize(points: np.ndarray, resolution: float, margin: int, sample: int) -> GridMeta:
    """포인트클라우드를 occupancy voxel grid로 변환한다.
    grid[ix, iy, iz] == 1 -> 장애물, 0 -> 자유 공간."""
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


def world_to_voxel(gm: GridMeta, p) -> tuple[int, int, int]:
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
    """v가 장애물 칸이면 가장 가까운 자유 공간 칸을 찾는다."""
    ix, iy, iz = v
    if grid_at(gm, ix, iy, iz) == 0:
        return v
    for radius in range(1, max_radius + 1):
        best, best_d = None, math.inf
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius and abs(dz) != radius:
                        continue
                    candidate = (ix + dx, iy + dy, iz + dz)
                    if grid_at(gm, *candidate) == 0:
                        d = dx * dx + dy * dy + dz * dz
                        if d < best_d:
                            best, best_d = candidate, d
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
    """장애물에 가리지 않는 waypoint는 건너뛰어 경로를 단순화한다 (string-pulling)."""
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


def path_length(path: list[np.ndarray] | None) -> float:
    if not path or len(path) < 2:
        return math.inf
    return float(sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))))


# ──────────────────────────────────────────────────────────────────────────
# A* 알고리즘
# ──────────────────────────────────────────────────────────────────────────

def astar(gm: GridMeta, start: np.ndarray, goal: np.ndarray, max_iters: int = 200_000):
    """26방향(3x3x3 이웃) A* 탐색. 반환: (path | None, info dict)"""
    sv = find_nearest_free(gm, world_to_voxel(gm, start))
    ev = find_nearest_free(gm, world_to_voxel(gm, goal))
    if sv is None or ev is None:
        return None, {"expanded": 0, "reason": "시작점 또는 목표점 근처에 자유 공간이 없음"}

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
            return None, {"expanded": expanded, "reason": "max_iters 초과"}

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

    return None, {"expanded": expanded, "reason": "경로를 찾지 못함 (도달 불가)"}


# ──────────────────────────────────────────────────────────────────────────
# rooms_graph.json 기반 계층적 A*
# ──────────────────────────────────────────────────────────────────────────

def _load_rooms_metadata(json_path: Path):
    """rooms_graph.json 로드 → (rooms_meta, adjacency, npy_to_room).
    - rooms_meta: room_id → {bbox_min, bbox_max, center, passages, npy}
    - adjacency: room_id → [(neighbor_room_id, door_center, edge_type), ...]
    - npy_to_room: npy 폴더명 → room_id
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    rooms_meta = {}
    for room_id, info in data.get("rooms", {}).items():
        rooms_meta[room_id] = {
            "bbox_min": np.array(info["bbox_min"], dtype=float),
            "bbox_max": np.array(info["bbox_max"], dtype=float),
            "center": np.array(info["center"], dtype=float),
            "passages": [np.array(p, dtype=float) for p in info.get("passages", [])],
            "npy": info["npy"],
        }

    adjacency = {rid: [] for rid in rooms_meta}
    for edge in data.get("edges", []):
        a, b = edge["a"], edge["b"]
        door = np.array(edge["door_center"], dtype=float)
        etype = edge.get("type", "door")
        if a in adjacency and b in adjacency:
            adjacency[a].append((b, door, etype))
            adjacency[b].append((a, door, etype))

    npy_to_room = {meta["npy"]: rid for rid, meta in rooms_meta.items()}
    return rooms_meta, adjacency, npy_to_room


def find_room_by_point(point, rooms_meta: dict) -> str | None:
    """point가 어느 방의 bbox에 들어가는지 찾는다.
    bbox 안에 들어가는 방이 없으면, 가장 가까운 방의 center를 기준으로 반환."""
    p = np.asarray(point, dtype=float)
    for room_id, meta in rooms_meta.items():
        if np.all(meta["bbox_min"] <= p) and np.all(p <= meta["bbox_max"]):
            return room_id

    best_id, best_d = None, math.inf
    for room_id, meta in rooms_meta.items():
        d = float(np.linalg.norm(meta["center"] - p))
        if d < best_d:
            best_id, best_d = room_id, d
    return best_id


def find_room_sequence(start_room: str, goal_room: str, adjacency: dict):
    """BFS로 시작 방 → 목표 방의 최단 방 시퀀스를 찾는다.
    반환: [(room_id, door_to_next | None), ...]  연결 불가면 None."""
    from collections import deque

    if start_room == goal_room:
        return [(start_room, None)]

    parent = {start_room: (None, None)}  # room → (prev_room, door)
    queue = deque([start_room])
    while queue:
        cur = queue.popleft()
        if cur == goal_room:
            break
        for nbr, door, _ in adjacency.get(cur, []):
            if nbr not in parent:
                parent[nbr] = (cur, door)
                queue.append(nbr)

    if goal_room not in parent:
        return None

    rooms_path = []
    cur = goal_room
    while cur is not None:
        rooms_path.append(cur)
        cur = parent[cur][0]
    rooms_path.reverse()

    # 각 방 → 다음 방으로 가는 door 좌표 매칭
    sequence = []
    for i, rid in enumerate(rooms_path):
        if i == len(rooms_path) - 1:
            sequence.append((rid, None))
        else:
            next_rid = rooms_path[i + 1]
            door = next(d for nbr, d, _ in adjacency[rid] if nbr == next_rid)
            sequence.append((rid, door))
    return sequence


def find_path_hierarchical(
    npy_dir: Path,
    rooms_json: Path,
    start,
    goal,
    resolution: float = 0.15,
    margin: int = 0,
    sample: int = 10,
    smooth: bool = True,
    verbose: bool = True,
):
    """rooms_graph.json을 활용한 계층적 A*.
    1) 시작/목표가 속한 방을 찾고
    2) edges 기반 BFS로 방 시퀀스를 찾고
    3) 각 방 안에서만 A* 실행 → 결과 이어붙임

    반환: (path, info dict, obstacle_points for 시각화)
    """
    rooms_meta, adjacency, _ = _load_rooms_metadata(rooms_json)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    start_room = find_room_by_point(start, rooms_meta)
    goal_room = find_room_by_point(goal, rooms_meta)
    if verbose:
        print(f"  시작점이 속한 방: {start_room}")
        print(f"  목표점이 속한 방: {goal_room}")
    if start_room is None or goal_room is None:
        raise ValueError("시작점/목표점이 어느 방에도 해당하지 않습니다.")

    sequence = find_room_sequence(start_room, goal_room, adjacency)
    if sequence is None:
        raise ValueError(f"방 그래프에서 {start_room} → {goal_room} 경로를 찾지 못했습니다 "
                         "(edges로 연결되지 않음).")
    if verbose:
        print(f"  방 시퀀스: {' → '.join(rid for rid, _ in sequence)}")

    full_path: list[np.ndarray] = []
    obstacle_points: list[np.ndarray] = []
    segment_info = []

    for i, (room_id, door_to_next) in enumerate(sequence):
        seg_start = start if i == 0 else sequence[i - 1][1]
        seg_goal = goal if i == len(sequence) - 1 else door_to_next

        meta = rooms_meta[room_id]
        coord_path = Path(npy_dir) / meta["npy"] / "coord.npy"
        if not coord_path.exists():
            raise FileNotFoundError(f"방 {room_id}의 coord.npy 없음: {coord_path}")

        points = _load_npy_file(coord_path)
        # 시각화용 다운샘플링된 obstacle points 누적
        obstacle_points.append(points[:: max(1, sample * 3)])

        gm = voxelize(points, resolution, margin, sample)
        seg_path, info = astar(gm, seg_start, seg_goal)
        if seg_path is None:
            raise RuntimeError(
                f"방 {room_id} 안에서 A* 실패: "
                f"{tuple(seg_start.round(2))} → {tuple(seg_goal.round(2))}, "
                f"이유: {info.get('reason', '?')}"
            )
        if smooth:
            seg_path = smooth_path(seg_path, gm)

        seg_len = path_length(seg_path)
        if verbose:
            print(f"  [{i + 1}/{len(sequence)}] 방 {room_id}: "
                  f"{len(seg_path)} waypoints, {seg_len:.2f}m ({info['expanded']:,} nodes)")
        segment_info.append({"room_id": room_id, "waypoints": len(seg_path), "distance_m": seg_len})

        # 이전 방의 끝점 = 이번 방의 시작점이라서 중복 제거
        if full_path:
            full_path.extend(seg_path[1:])
        else:
            full_path.extend(seg_path)

    obstacles = np.concatenate(obstacle_points, axis=0) if obstacle_points else np.empty((0, 3))
    info = {
        "rooms_traversed": [rid for rid, _ in sequence],
        "n_rooms": len(sequence),
        "distance_m": path_length(full_path),
        "waypoints": len(full_path),
        "segments": segment_info,
    }
    return full_path, info, obstacles


# ──────────────────────────────────────────────────────────────────────────
# 결과 저장 / 시각화
# ──────────────────────────────────────────────────────────────────────────

def save_path(path: list[np.ndarray], output: Path) -> None:
    suffix = output.suffix.lower()
    arr = np.array(path)

    if suffix == ".npy":
        np.save(output, arr)
    elif suffix == ".csv":
        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y", "z"])
            writer.writerows(arr.tolist())
    elif suffix == ".json":
        with open(output, "w") as f:
            json.dump({"waypoints": arr.tolist()}, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError("저장 형식은 .npy, .csv, .json 중 하나여야 합니다.")


def plot_path(obstacle_points: np.ndarray, path: list[np.ndarray],
              start: np.ndarray, goal: np.ndarray, output: Path,
              tube_radius_hint: float = 0.06) -> None:
    """경로 + 장애물을 GLB(glTF binary)로 저장.
    웹 뷰어(예: https://gltf-viewer.donmccurdy.com/), macOS Preview, Blender, Three.js 등에서
    인터랙티브하게 회전/줌하며 볼 수 있다.

    obstacle_points: (N, 3) 회색으로 그릴 장애물 포인트들의 world 좌표.
    """
    try:
        import trimesh
    except ImportError as e:
        raise ImportError(
            "GLB 시각화를 위해 trimesh가 필요합니다. `pip install trimesh`로 설치하세요."
        ) from e

    scene = trimesh.Scene()

    # 1) 장애물 포인트클라우드 (회색)
    if len(obstacle_points) > 0:
        pts = np.asarray(obstacle_points)
        if len(pts) > 80000:  # 너무 많으면 다운샘플링
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pts), 80000, replace=False)
            pts = pts[idx]
        colors = np.tile(np.array([170, 170, 170, 200], dtype=np.uint8), (len(pts), 1))
        scene.add_geometry(trimesh.PointCloud(pts, colors=colors), geom_name="obstacles")

    path_arr = np.array(path)

    # 2) 경로 (파란색 cylinder들)
    diag = float(np.linalg.norm(path_arr[-1] - path_arr[0]))
    tube_radius = max(diag * 0.005, tube_radius_hint)
    z_axis = np.array([0.0, 0.0, 1.0])

    for i in range(len(path_arr) - 1):
        a, b = path_arr[i], path_arr[i + 1]
        seg_vec = b - a
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=tube_radius, height=seg_len, sections=16)
        direction = seg_vec / seg_len
        rot_axis = np.cross(z_axis, direction)
        rot_axis_norm = float(np.linalg.norm(rot_axis))
        if rot_axis_norm > 1e-6:
            rot_axis /= rot_axis_norm
            angle = math.acos(float(np.clip(np.dot(z_axis, direction), -1.0, 1.0)))
            R = trimesh.transformations.rotation_matrix(angle, rot_axis)
        else:
            R = trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0]) if direction[2] < 0 else np.eye(4)
        T = trimesh.transformations.translation_matrix((a + b) / 2)
        cyl.apply_transform(T @ R)
        cyl.visual.face_colors = [0, 148, 212, 255]
        scene.add_geometry(cyl, geom_name=f"path_seg_{i:03d}")

    # 3) waypoint sphere (작은 파란색 구)
    for i, p in enumerate(path_arr[1:-1], start=1):
        sph = trimesh.creation.icosphere(radius=tube_radius * 1.5, subdivisions=2)
        sph.apply_translation(p)
        sph.visual.face_colors = [0, 148, 212, 255]
        scene.add_geometry(sph, geom_name=f"waypoint_{i:03d}")

    # 4) 시작점 (녹색) / 목표점 (빨간색)
    marker_r = tube_radius * 4
    start_sphere = trimesh.creation.icosphere(radius=marker_r, subdivisions=3)
    start_sphere.apply_translation(start)
    start_sphere.visual.face_colors = [40, 200, 80, 255]
    scene.add_geometry(start_sphere, geom_name="start")

    goal_sphere = trimesh.creation.icosphere(radius=marker_r, subdivisions=3)
    goal_sphere.apply_translation(goal)
    goal_sphere.visual.face_colors = [230, 60, 60, 255]
    scene.add_geometry(goal_sphere, geom_name="goal")

    scene.export(output)


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def find_path(
    input_path: Path,
    start,
    goal,
    resolution: float = 0.15,
    margin: int = 0,
    sample: int = 10,
    smooth: bool = True,
):
    """다른 스크립트에서 import해서 쓸 수 있는 함수형 인터페이스.
    반환: (path: list[np.ndarray] | None, info: dict, gm: GridMeta)"""
    points = load_points(input_path)
    gm = voxelize(points, resolution, margin, sample)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)

    path, info = astar(gm, start, goal)
    if path is not None and smooth:
        path = smooth_path(path, gm)
    info["distance_m"] = path_length(path)
    info["waypoints"] = len(path) if path else 0
    return path, info, gm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D 포인트클라우드 위에서 A*로 시작점→목표점 경로를 찾습니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="포인트클라우드 입력: .npy/.ply 파일, coord.npy가 있는 방 폴더, "
                             "또는 여러 방 폴더의 상위 폴더(npy/)")
    parser.add_argument("--rooms-json", type=Path, default=None,
                        help="rooms_graph.json. 지정 시 계층적 A* 모드: 방 그래프 BFS로 방 시퀀스를 찾고 "
                             "각 방 안에서만 A* 실행 → 결과 이어붙임. (--input은 npy 상위 폴더여야 함)")
    parser.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="시작점 좌표")
    parser.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"), help="목표점 좌표")
    parser.add_argument("--resolution", type=float, default=0.15, help="voxel 한 칸의 크기 (m)")
    parser.add_argument("--margin", type=int, default=0, help="장애물 팽창 마진 (voxel 단위)")
    parser.add_argument("--sample", type=int, default=10, help="포인트클라우드 다운샘플링 간격 (1=전체 사용)")
    parser.add_argument("--no-smooth", action="store_true", help="경로 단순화(string-pulling) 끄기")
    parser.add_argument("--save", type=Path, default=None, help="경로 저장 파일 (.npy/.csv/.json)")
    parser.add_argument("--plot", action="store_true", help="3D 경로를 GLB 파일로 저장 (웹 뷰어/Blender/macOS Preview에서 인터랙티브 시청)")
    parser.add_argument("--plot-output", type=Path, default=Path("astar_path.glb"), help="GLB 시각화 저장 경로")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"[에러] 입력 파일을 찾을 수 없습니다: {args.input}")

    start = np.asarray(args.start, dtype=float)
    goal = np.asarray(args.goal, dtype=float)

    import time
    t0 = time.perf_counter()

    if args.rooms_json is not None:
        # ─── 계층적 모드 (rooms_graph.json 활용) ────────────────────
        if not args.rooms_json.exists():
            raise SystemExit(f"[에러] rooms-json 파일 없음: {args.rooms_json}")
        if not args.input.is_dir():
            raise SystemExit(f"[에러] 계층적 모드에서 --input은 npy 상위 폴더여야 합니다.")

        print(f"입력 폴더: {args.input}")
        print(f"방 그래프: {args.rooms_json}")
        print(f"시작점: {tuple(start)}  목표점: {tuple(goal)}\n")

        try:
            path, info, obstacles = find_path_hierarchical(
                args.input, args.rooms_json, start, goal,
                resolution=args.resolution, margin=args.margin, sample=args.sample,
                smooth=not args.no_smooth,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"\n[실패] {e}")
            raise SystemExit(1)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        print(f"\n[성공] 계층적 A* 경로 탐색 완료")
        print(f"  총 소요 시간: {elapsed_ms:.1f} ms")
        print(f"  방 시퀀스:    {' → '.join(info['rooms_traversed'])}")
        print(f"  총 경로 길이: {info['distance_m']:.2f} m")
        print(f"  Waypoint 수:  {info['waypoints']}")

    else:
        # ─── 단순 모드 (모든 포인트 병합) ───────────────────────────
        print(f"입력 로드 중: {args.input}")
        points = load_points(args.input)
        print(f"  포인트 수: {len(points):,}")

        gm = voxelize(points, args.resolution, args.margin, args.sample)
        print(f"격자 크기: {gm.shape}, 자유 공간 비율: {gm.free_ratio * 100:.1f}%")
        print(f"시작점: {tuple(start)}  목표점: {tuple(goal)}")

        path, search_info = astar(gm, start, goal)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if path is None:
            print(f"\n[실패] 경로를 찾지 못했습니다. ({search_info.get('reason', '?')})")
            print(f"  탐색한 노드 수: {search_info.get('expanded', 0):,}")
            raise SystemExit(1)

        if not args.no_smooth:
            path = smooth_path(path, gm)

        print(f"\n[성공] 경로 탐색 완료")
        print(f"  소요 시간:    {elapsed_ms:.1f} ms")
        print(f"  탐색 노드 수: {search_info['expanded']:,}")
        print(f"  경로 길이:    {path_length(path):.2f} m")
        print(f"  Waypoint 수:  {len(path)}")

        # 시각화용 obstacle points (gm에서 추출)
        occ = np.argwhere(gm.grid == 1)
        obstacles = occ * gm.resolution + np.array([gm.x_min, gm.y_min, gm.z_min]) if len(occ) > 0 else np.empty((0, 3))

    # 저장 / 시각화
    if args.save:
        save_path(path, args.save)
        print(f"  저장됨:       {args.save.resolve()}")

    if args.plot:
        plot_path(obstacles, path, start, goal, args.plot_output,
                  tube_radius_hint=max(args.resolution * 0.4, 0.05))
        print(f"  시각화 저장:  {args.plot_output.resolve()}")


if __name__ == "__main__":
    main()
