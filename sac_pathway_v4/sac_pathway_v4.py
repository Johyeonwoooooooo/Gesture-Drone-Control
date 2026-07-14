#!/usr/bin/env python3
"""
sac_pathway_v4.py — 관측/환경 layer 개선 SAC 경로 탐색.

sac_pathway_v3.py 대비 개선 사항 (v3의 α 고착, rollback 9회, curriculum 8.8m 정체
문제의 근본 원인이 관측·환경 설계에 있다는 진단 → 3가지 대응):

  1) 환경 안정화:
     - 연속 충돌 조기 종료 (--max-consec-collisions, 기본 5회)
     - 에피소드당 총 충돌 페널티 상한 (--collision-cap, 기본 30)
     → -1000 return 폭주 제거, replay 오염 완화

  2) 관측 강화 (부분 관측 문제 완화):
     - Raycast range 3m → 5m (--ray-range 기본 5.0)
     - Ray 밀도 26 → ~98 방향 (--ray-density 2 로 5x5x5 primitive)
     - Frame stacking: 최근 3 스텝의 rays + 최근 2 스텝의 이전 행동 스택
     → 부분 관측(POMDP) 완화, 코너 회피 능력 및 속도/가속 정보 확보

  3) Target entropy annealing:
     - target_entropy: -|A|=-3 (초반) → -6 (후반)
     - 학습 진행률의 앞 70%에 걸쳐 선형 감소 (--target-entropy-frac)
     → α가 0.10 고착에서 벗어나 0.03~0.05까지 내려가며 결정적 policy로

v3에서 가져온 것: SAC v3(auto-α + LayerNorm), HER, n-step, PER,
                adaptive curriculum, best-model rollback, 계층적 planning, Open3D.

사용 예:
  python sac_pathway_v4.py train --input npy/ --total-steps 300000
  python sac_pathway_v4.py plan  --input npy/ --rooms-json rooms_graph.json \
      --model ~/Desktop/sac_pathway_v4/sac_model_best.pt \
      --start 4.4 0.1 -1.0 --goal 11.2 5.0 4.3 --show
  python sac_pathway_v4.py curve

산출물은 기본적으로 ~/Desktop/sac_pathway_v4/에 저장된다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import struct
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# 격자(voxel grid) — pathway.py와 동일한 로직
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


def load_points(path: Path) -> np.ndarray:
    if path.is_dir():
        return _load_points_from_dir(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _load_npy_file(path)
    if suffix == ".ply":
        return _load_ply_points(path)
    raise ValueError(f"지원하지 않는 입력 형식입니다: {suffix} (.npy, .ply, 또는 폴더만 지원)")


def _load_npy_file(path: Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    if arr.shape[1] < 3:
        raise ValueError(f"{path}: 최소 3개 컬럼(x, y, z)을 포함해야 합니다.")
    return arr[:, :3]


def _load_points_from_dir(path: Path) -> np.ndarray:
    direct_coord = path / "coord.npy"
    if direct_coord.exists():
        print(f"  방 폴더로 인식: {path.name}/coord.npy 로드")
        return _load_npy_file(direct_coord)

    room_dirs = sorted(d for d in path.iterdir() if d.is_dir() and (d / "coord.npy").exists())
    if not room_dirs:
        raise ValueError(f"{path} 안에서 coord.npy를 찾을 수 없습니다.")
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

    sizes = {"char": 1, "uchar": 1, "int8": 1, "uint8": 1, "short": 2, "ushort": 2,
             "int16": 2, "uint16": 2, "int": 4, "uint": 4, "int32": 4, "uint32": 4,
             "float": 4, "float32": 4, "double": 8, "float64": 8}
    formats = {"char": "b", "uchar": "B", "int8": "b", "uint8": "B", "short": "h",
               "ushort": "H", "int16": "h", "uint16": "H", "int": "i", "uint": "I",
               "int32": "i", "uint32": "I", "float": "f", "float32": "f",
               "double": "d", "float64": "d"}
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


def world_to_voxel(gm: GridMeta, p) -> tuple[int, int, int]:
    x, y, z = p
    return (round((x - gm.x_min) / gm.resolution),
            round((y - gm.y_min) / gm.resolution),
            round((z - gm.z_min) / gm.resolution))


def voxel_to_world(gm: GridMeta, v) -> np.ndarray:
    ix, iy, iz = v
    return np.array([gm.x_min + ix * gm.resolution,
                     gm.y_min + iy * gm.resolution,
                     gm.z_min + iz * gm.resolution], dtype=float)


def grid_at(gm: GridMeta, ix: int, iy: int, iz: int) -> int:
    nx, ny, nz = gm.shape
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return 1
    return int(gm.grid[ix, iy, iz])


def is_free_world(gm: GridMeta, p) -> bool:
    return grid_at(gm, *world_to_voxel(gm, p)) == 0


def find_nearest_free(gm: GridMeta, v: tuple[int, int, int], max_radius: int = 20):
    """v가 장애물 칸이면 가장 가까운 자유 공간 칸을 찾는다 (pathway.py와 동일)."""
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


def snap_to_free(gm: GridMeta, p: np.ndarray) -> np.ndarray:
    """world 좌표 p를 가장 가까운 자유 voxel 중심으로 스냅."""
    v = find_nearest_free(gm, world_to_voxel(gm, p))
    return voxel_to_world(gm, v) if v is not None else np.asarray(p, dtype=float)


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


def path_length(path: list[np.ndarray] | None) -> float:
    if not path or len(path) < 2:
        return math.inf
    return float(sum(np.linalg.norm(path[i] - path[i - 1]) for i in range(1, len(path))))


# ──────────────────────────────────────────────────────────────────────────
# rooms_graph.json — 방 그래프 (pathway.py의 계층적 A*와 동일한 구조)
# ──────────────────────────────────────────────────────────────────────────

def load_rooms_metadata(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)

    rooms_meta = {}
    for room_id, info in data.get("rooms", {}).items():
        rooms_meta[room_id] = {
            "bbox_min": np.array(info["bbox_min"], dtype=float),
            "bbox_max": np.array(info["bbox_max"], dtype=float),
            "center": np.array(info["center"], dtype=float),
            "floor": info.get("floor", 0),
        }

    adjacency = {rid: [] for rid in rooms_meta}
    for edge in data.get("edges", []):
        a, b = edge["a"], edge["b"]
        door = np.array(edge["door_center"], dtype=float)
        etype = edge.get("type", "door")
        if a in adjacency and b in adjacency:
            adjacency[a].append((b, door, etype))
            adjacency[b].append((a, door, etype))
    return rooms_meta, adjacency


def find_room_by_point(point, rooms_meta: dict) -> str | None:
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
    """BFS로 방 시퀀스와 각 방→다음 방으로 가는 door 좌표를 찾는다."""
    if start_room == goal_room:
        return [(start_room, None)]

    parent = {start_room: (None, None)}
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

    sequence = []
    for i, rid in enumerate(rooms_path):
        if i == len(rooms_path) - 1:
            sequence.append((rid, None))
        else:
            next_rid = rooms_path[i + 1]
            door = next(d for nbr, d, _ in adjacency[rid] if nbr == next_rid)
            sequence.append((rid, door))
    return sequence


def make_subgoals(start, goal, rooms_json: Path, gm: GridMeta, verbose: bool = True):
    """rooms_graph 기반으로 [door1, door2, ..., goal] subgoal 리스트 생성.
    door 좌표는 자유 공간으로 스냅한다."""
    rooms_meta, adjacency = load_rooms_metadata(rooms_json)
    start_room = find_room_by_point(start, rooms_meta)
    goal_room = find_room_by_point(goal, rooms_meta)
    if verbose:
        print(f"  시작점이 속한 방: {start_room} (floor {rooms_meta[start_room]['floor']})")
        print(f"  목표점이 속한 방: {goal_room} (floor {rooms_meta[goal_room]['floor']})")

    sequence = find_room_sequence(start_room, goal_room, adjacency)
    if sequence is None:
        raise ValueError(f"방 그래프에서 {start_room} → {goal_room} 경로 없음 (edges 미연결)")
    if verbose:
        print(f"  방 시퀀스: {' → '.join(rid for rid, _ in sequence)}")

    subgoals = []
    for rid, door in sequence:
        if door is not None:
            subgoals.append(snap_to_free(gm, door))
    subgoals.append(np.asarray(goal, dtype=float))
    return subgoals, [rid for rid, _ in sequence]


# ──────────────────────────────────────────────────────────────────────────
# RL 환경 (v1과 동일한 MDP + HER/커리큘럼을 위한 확장)
# ──────────────────────────────────────────────────────────────────────────

RAY_DIRS = np.array(
    [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if not (dx == dy == dz == 0)],
    dtype=float,
)
RAY_DIRS /= np.linalg.norm(RAY_DIRS, axis=1, keepdims=True)


def make_ray_dirs(density: int = 1) -> np.ndarray:
    """v4: 밀도별 raycast 방향 생성.

    density=1: 3x3x3 큐브의 이웃 26방향 (v1~v3와 동일)
    density=2: 5x5x5 큐브의 primitive 방향 (~98개). (2,1,0)/(2,1,1)/(2,2,1) 같은
              중간 각도가 추가되어 좁은 문/코너 통과 능력 향상.
    """
    if density < 1:
        raise ValueError("density는 1 이상")
    offsets = [(dx, dy, dz)
               for dx in range(-density, density + 1)
               for dy in range(-density, density + 1)
               for dz in range(-density, density + 1)
               if not (dx == dy == dz == 0)]
    dirs: list[np.ndarray] = []
    for o in offsets:
        v = np.array(o, dtype=float)
        v /= np.linalg.norm(v)
        # 정규화 후 중복 방향 제거 (예: (2,0,0)과 (1,0,0)은 같은 방향)
        if not any(np.allclose(v, existing, atol=1e-6) for existing in dirs):
            dirs.append(v)
    return np.stack(dirs)


class PointCloudNavEnv:
    """goal-conditioned 3D 네비게이션 환경 (v4).

    v4 변경:
      - Ray density 파라미터 (26 → ~98 방향)
      - Frame stacking: 최근 N 스텝 rays + 최근 N-1 스텝 이전 행동
      - 연속 충돌 조기 종료
      - 에피소드당 충돌 페널티 상한

    관측 구조 (frame_stack=3, ray_density=1 기준: 3+1+3*26+2*3 = 88차원):
      [0:3]           rel = (goal - pos) / diag         ← 현재 목표 상대 벡터
      [3]             dist = |goal - pos| / diag        ← 현재 목표 거리
      [4:4+3*N]       rays_stack (t, t-1, t-2)         ← N방향 raycast × 3프레임
      [4+3*N:end]     action_stack (t-1, t-2)          ← 이전 행동 × 2프레임
    """

    ACT_DIM = 3

    def __init__(self, gm: GridMeta, max_step: float | None = None,
                 max_episode_steps: int = 500, goal_threshold: float | None = None,
                 ray_range: float = 5.0, ray_density: int = 1, frame_stack: int = 3,
                 max_consec_collisions: int = 5, collision_cap: float = 30.0,
                 progress_coef: float = 10.0, time_penalty: float = 0.05,
                 collision_penalty: float = 2.0, goal_reward: float = 100.0,
                 seed: int = 0):
        self.gm = gm
        self.max_step = max_step or gm.resolution * 2.0
        self.max_episode_steps = max_episode_steps
        self.goal_threshold = goal_threshold or gm.resolution * 2.0
        self.ray_range = ray_range
        self.frame_stack = max(1, frame_stack)
        self.max_consec_collisions = max_consec_collisions
        self.collision_cap = collision_cap
        self.progress_coef = progress_coef
        self.time_penalty = time_penalty
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.rng = np.random.default_rng(seed)

        # v4: 방향 집합을 인스턴스 속성으로
        self.ray_dirs = make_ray_dirs(ray_density)
        self.n_rays = len(self.ray_dirs)

        # 관측 차원 (인스턴스별)
        self.OBS_DIM = 4 + self.frame_stack * self.n_rays + (self.frame_stack - 1) * self.ACT_DIM

        nx, ny, nz = gm.shape
        self.scene_diag = float(np.linalg.norm(np.array([nx, ny, nz]) * gm.resolution))
        self.free_voxels = np.argwhere(gm.grid == 0)
        if len(self.free_voxels) == 0:
            raise ValueError("자유 공간 voxel이 없습니다. resolution/sample을 확인하세요.")

        self.pos = np.zeros(3)
        self.goal = np.zeros(3)
        self.steps = 0
        self.prev_dist = 0.0
        # 프레임 스택 버퍼: 최근 것이 index 0
        self.rays_stack = np.ones((self.frame_stack, self.n_rays), dtype=np.float32)
        self.action_stack = np.zeros((self.frame_stack - 1, self.ACT_DIM), dtype=np.float32)
        # v4: 충돌 안전장치
        self.consecutive_collisions = 0
        self.episode_collision_total = 0.0

    def _sample_free_point(self) -> np.ndarray:
        idx = self.rng.integers(0, len(self.free_voxels))
        v = self.free_voxels[idx]
        return np.array([self.gm.x_min, self.gm.y_min, self.gm.z_min]) + v * self.gm.resolution

    def reset(self, start: np.ndarray | None = None, goal: np.ndarray | None = None,
              min_dist: float = 1.0, max_goal_dist: float | None = None) -> np.ndarray:
        if start is not None and goal is not None:
            self.pos = np.asarray(start, dtype=float).copy()
            self.goal = np.asarray(goal, dtype=float).copy()
        else:
            self.pos = self._sample_free_point()
            self.goal = self._sample_free_point()
            for _ in range(200):
                g = self._sample_free_point()
                d = np.linalg.norm(g - self.pos)
                if d < min_dist:
                    continue
                if max_goal_dist is not None and d > max_goal_dist:
                    continue
                self.goal = g
                break
        self.steps = 0
        self.prev_dist = float(np.linalg.norm(self.goal - self.pos))
        # 프레임 스택 초기화: 현재 rays로 모두 채우고 행동 스택은 0
        current_rays = self._raycast(self.pos)
        self.rays_stack = np.tile(current_rays, (self.frame_stack, 1)).astype(np.float32)
        self.action_stack = np.zeros((self.frame_stack - 1, self.ACT_DIM), dtype=np.float32)
        # 충돌 안전장치 리셋
        self.consecutive_collisions = 0
        self.episode_collision_total = 0.0
        return self.build_obs(self.pos, self.goal, self.rays_stack, self.action_stack)

    # ── 관측 ─────────────────────────────────────────────────────────
    def _raycast(self, pos: np.ndarray) -> np.ndarray:
        dists = np.ones(self.n_rays, dtype=np.float32)
        step = self.gm.resolution * 0.5
        n_steps = int(self.ray_range / step)
        for i, d in enumerate(self.ray_dirs):
            for k in range(1, n_steps + 1):
                p = pos + d * (k * step)
                if grid_at(self.gm, *world_to_voxel(self.gm, p)) != 0:
                    dists[i] = (k * step) / self.ray_range
                    break
        return dists

    def build_obs(self, pos: np.ndarray, goal: np.ndarray,
                  rays_stack: np.ndarray, action_stack: np.ndarray) -> np.ndarray:
        """관측 재구성. rays/action stack은 goal 무관 → HER 재라벨링 시 그대로 재사용."""
        rel = (goal - pos) / self.scene_diag
        dist = np.linalg.norm(goal - pos) / self.scene_diag
        return np.concatenate([
            rel, [dist],
            rays_stack.flatten(),
            action_stack.flatten(),
        ]).astype(np.float32)

    # ── 보상 (HER 재라벨링과 공식 공유) ──────────────────────────────
    def compute_reward(self, prev_pos, new_pos, goal, collided: bool,
                       collision_budget_left: float | None = None):
        """반환: (reward, success). v4: 충돌 페널티는 남은 budget까지만 부과."""
        reward = -self.time_penalty
        if collided:
            pen = self.collision_penalty
            if collision_budget_left is not None:
                pen = max(0.0, min(pen, collision_budget_left))
            reward -= pen
        prev_d = float(np.linalg.norm(goal - prev_pos))
        new_d = float(np.linalg.norm(goal - new_pos))
        reward += self.progress_coef * (prev_d - new_d)
        success = new_d <= self.goal_threshold
        if success:
            reward += self.goal_reward
        return reward, success

    # ── 전이 ─────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        self.steps += 1
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0).astype(np.float32)
        prev_pos = self.pos.copy()
        new_pos = self.pos + action * self.max_step

        collided = False
        if is_free_world(self.gm, new_pos) and line_of_sight(self.gm, self.pos, new_pos, steps=4):
            self.pos = new_pos
            self.consecutive_collisions = 0
        else:
            collided = True
            self.consecutive_collisions += 1

        budget_left = max(0.0, self.collision_cap - self.episode_collision_total)
        reward, success = self.compute_reward(prev_pos, self.pos, self.goal, collided,
                                              collision_budget_left=budget_left)
        if collided:
            self.episode_collision_total += min(self.collision_penalty, budget_left)

        self.prev_dist = float(np.linalg.norm(self.goal - self.pos))
        current_rays = self._raycast(self.pos)

        # 프레임 스택 shift: 오래된 것 뒤로, 새 것 앞으로
        self.rays_stack = np.roll(self.rays_stack, 1, axis=0)
        self.rays_stack[0] = current_rays
        if self.frame_stack > 1:
            self.action_stack = np.roll(self.action_stack, 1, axis=0)
            self.action_stack[0] = action

        # v4 조기 종료: 연속 충돌 임계값 초과
        stuck = self.consecutive_collisions >= self.max_consec_collisions
        done = success or stuck or self.steps >= self.max_episode_steps
        info = {"success": success, "collided": collided, "stuck": stuck,
                "dist": self.prev_dist, "prev_pos": prev_pos,
                "rays": current_rays,
                # HER용 스냅샷 (전이 이전 스택)
                "rays_stack_before": None, "action_stack_before": None}
        return self.build_obs(self.pos, self.goal, self.rays_stack,
                              self.action_stack), reward, done, info


# ──────────────────────────────────────────────────────────────────────────
# SAC v2 (자동 온도 조정 — Haarnoja et al. 2019)
#   v1과의 차이: V 네트워크 대신 twin target Q 사용, α를 학습으로 자동 조정.
#   target: y = r + γ(1-d)·[min Q_targ(s', ã') - α·log π(ã'|s')]
#   α loss: J(α) = E[-α·(log π(a|s) + H_target)],  H_target = -|A|
# ──────────────────────────────────────────────────────────────────────────

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
EPS = 1e-6


def mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    """v3: 각 hidden 뒤에 LayerNorm 추가 (Nauman et al. 2024).
    Q overestimation을 근본적으로 완화 — v2의 붕괴를 잡는 핵심."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        # v3: policy에도 LayerNorm 추가
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, act_dim)
        self.log_std_head = nn.Linear(256, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        a = torch.tanh(u)
        log_prob = normal.log_prob(u).sum(-1) - torch.log(1 - a.pow(2) + EPS).sum(-1)
        return a, log_prob, torch.tanh(mean)


class SACv4Agent:
    """SAC v2(auto-α) + LayerNorm critic/policy + PER-compatible update (returns TD error).

    v2 대비 update signature 변화:
      - is_weights: PER의 importance-sampling 가중치 (uniform이면 1)
      - 반환값에 'td_error' 포함: PER 우선순위 갱신용
    """

    def __init__(self, obs_dim: int, act_dim: int, gamma: float = 0.99,
                 tau: float = 0.005, lr: float = 3e-4, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.gamma, self.tau = gamma, tau

        self.policy = GaussianPolicy(obs_dim, act_dim).to(self.device)
        self.q1 = mlp(obs_dim + act_dim, 1).to(self.device)
        self.q2 = mlp(obs_dim + act_dim, 1).to(self.device)
        self.q1_target = mlp(obs_dim + act_dim, 1).to(self.device)
        self.q2_target = mlp(obs_dim + act_dim, 1).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # v4: target entropy annealing
        #   초반: -|A|=-3 (충분 탐험) → 후반: target_entropy_end=-6 (결정적 policy)
        self.target_entropy_start = -float(act_dim)
        self.target_entropy_end = -float(act_dim)  # 기본은 anneal 없음
        self.target_entropy = self.target_entropy_start
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        self.policy_opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    def set_target_entropy_schedule(self, end: float, act_dim: int):
        """[v4] target_entropy end 값 설정. 실제 anneal은 매 update마다 진행률로 계산."""
        self.target_entropy_end = float(end)

    def update_target_entropy(self, progress: float) -> None:
        """[v4] 진행률 0~1을 받아 target_entropy를 선형 보간."""
        progress = float(np.clip(progress, 0.0, 1.0))
        self.target_entropy = (self.target_entropy_start
                               + progress * (self.target_entropy_end
                                             - self.target_entropy_start))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _, mean_a = self.policy.sample(obs_t)
        return (mean_a if deterministic else a).squeeze(0).cpu().numpy()

    def update(self, batch, is_weights: np.ndarray | None = None,
               n_step: int = 1):
        """SAC v3 gradient step.

        Args:
          batch: (s, a, r, s2, d) — r은 n-step 누적 보상, d는 n-step 종단 플래그.
          is_weights: PER importance-sampling 가중치 (uniform이면 None).
          n_step: n-step return의 n (γ^n을 target에 사용).
        """
        s, a, r, s2, d = (torch.as_tensor(x, dtype=torch.float32, device=self.device)
                          for x in batch)
        if is_weights is None:
            w = torch.ones_like(r)
        else:
            w = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)
        alpha = self.alpha.detach()
        gamma_n = self.gamma ** n_step  # n-step bootstrap 계수

        # ── Q 업데이트: y = r_bar + γ^n(1-d)[min Q_targ(s',ã') - α log π(ã'|s')] ──
        with torch.no_grad():
            a2, log_prob2, _ = self.policy.sample(s2)
            sa2 = torch.cat([s2, a2], dim=-1)
            q_targ = torch.min(self.q1_target(sa2), self.q2_target(sa2)).squeeze(-1)
            y = r + gamma_n * (1.0 - d) * (q_targ - alpha * log_prob2)
        sa = torch.cat([s, a], dim=-1)
        q1_pred = self.q1(sa).squeeze(-1)
        q2_pred = self.q2(sa).squeeze(-1)
        # IS 가중 MSE (PER 편향 보정)
        q1_loss = (w * (q1_pred - y).pow(2)).mean()
        q2_loss = (w * (q2_pred - y).pow(2)).mean()
        self.q1_opt.zero_grad(); q1_loss.backward(); self.q1_opt.step()
        self.q2_opt.zero_grad(); q2_loss.backward(); self.q2_opt.step()

        # PER 우선순위: 두 Q의 평균 TD-error
        td_error = ((q1_pred + q2_pred) / 2 - y).detach().abs().cpu().numpy()

        # ── Policy 업데이트: J_π = E[α log π(ã|s) - min Q(s,ã)] ──
        a_pi, log_prob_pi, _ = self.policy.sample(s)
        sa_pi = torch.cat([s, a_pi], dim=-1)
        q_min_pi = torch.min(self.q1(sa_pi), self.q2(sa_pi)).squeeze(-1)
        policy_loss = (alpha * log_prob_pi - q_min_pi).mean()
        self.policy_opt.zero_grad(); policy_loss.backward(); self.policy_opt.step()

        # ── 온도 α 업데이트: J(α) = E[-α(log π + H_target)] ──
        alpha_loss = -(self.log_alpha * (log_prob_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

        # ── target Q soft update ──
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)

        return {"q1_loss": q1_loss.item(), "policy_loss": policy_loss.item(),
                "alpha": float(alpha), "td_error": td_error}

    def state_snapshot(self) -> dict:
        """Best-model rollback용: 현재 state를 CPU 텐서로 스냅샷."""
        return {
            "policy": {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()},
            "q1": {k: v.detach().cpu().clone() for k, v in self.q1.state_dict().items()},
            "q2": {k: v.detach().cpu().clone() for k, v in self.q2.state_dict().items()},
            "q1_target": {k: v.detach().cpu().clone() for k, v in self.q1_target.state_dict().items()},
            "q2_target": {k: v.detach().cpu().clone() for k, v in self.q2_target.state_dict().items()},
            "log_alpha": self.log_alpha.detach().cpu().clone(),
        }

    def load_snapshot(self, snap: dict) -> None:
        """Best-model rollback: 스냅샷 복원."""
        self.policy.load_state_dict({k: v.to(self.device) for k, v in snap["policy"].items()})
        self.q1.load_state_dict({k: v.to(self.device) for k, v in snap["q1"].items()})
        self.q2.load_state_dict({k: v.to(self.device) for k, v in snap["q2"].items()})
        self.q1_target.load_state_dict({k: v.to(self.device) for k, v in snap["q1_target"].items()})
        self.q2_target.load_state_dict({k: v.to(self.device) for k, v in snap["q2_target"].items()})
        with torch.no_grad():
            self.log_alpha.copy_(snap["log_alpha"].to(self.device))

    def save(self, path: Path, config: dict):
        torch.save({
            "version": "v4",
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "config": config,
        }, path)

    @classmethod
    def load(cls, path: Path, obs_dim: int, act_dim: int) -> tuple["SACv4Agent", dict]:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt.get("version") != "v4":
            raise SystemExit(f"[에러] v4 모델이 아닙니다 (version={ckpt.get('version')}). "
                             "v4로 다시 학습하세요.")
        agent = cls(obs_dim, act_dim)
        agent.policy.load_state_dict(ckpt["policy"])
        agent.q1.load_state_dict(ckpt["q1"]); agent.q2.load_state_dict(ckpt["q2"])
        agent.q1_target.load_state_dict(ckpt["q1_target"])
        agent.q2_target.load_state_dict(ckpt["q2_target"])
        with torch.no_grad():
            agent.log_alpha.copy_(ckpt["log_alpha"].to(agent.device))
        return agent, ckpt.get("config", {})


class PrioritizedReplayBuffer:
    """PER (Schaul et al. 2016) + n-step returns.

    - 우선순위 p_i = (|TD_i| + ε)^α, 샘플링 확률 P(i) = p_i / Σp
    - importance-sampling 가중치 w_i = (N·P(i))^-β, max w로 정규화 (분산 억제)
    - 새 transition은 최대 우선순위로 push (즉시 학습 대상이 됨)

    n-step은 (s_t, a_t, r_bar, s_{t+n}, done)의 r_bar/done을 사용자가 직접
    계산해서 push. 이 버퍼는 저장 매체만 담당.
    """

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000,
                 alpha: float = 0.6, beta_start: float = 0.4,
                 beta_end: float = 1.0, beta_frames: int = 300_000,
                 epsilon: float = 1e-3):
        self.capacity = capacity
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float64)

        self.alpha = alpha           # 우선순위 스케일 (0=uniform, 1=fully prioritized)
        self.beta_start = beta_start # IS 보정 강도 초기값
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon       # priority 하한 (0 방지)

        self.idx = 0
        self.size = 0
        self.max_priority = 1.0
        self.frame = 0

    def push(self, s, a, r, s2, d):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(d)
        # 새 transition은 즉시 학습 대상이 되도록 최대 우선순위로
        self.priorities[i] = self.max_priority
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _beta(self) -> float:
        """β를 학습 진행에 따라 선형 증가 (annealing)."""
        frac = min(1.0, self.frame / max(1, self.beta_frames))
        return self.beta_start + (self.beta_end - self.beta_start) * frac

    def sample(self, batch_size: int):
        """Returns (batch, indices, is_weights).
        batch = (s, a, r, s2, d), indices는 update_priorities용."""
        self.frame += 1
        prios = self.priorities[:self.size] ** self.alpha
        probs = prios / prios.sum()
        idx = np.random.choice(self.size, size=batch_size, p=probs, replace=True)

        # IS 가중치 (Schaul et al. Eq. 1): w_i = (1/N · 1/P(i))^β, max로 정규화
        beta = self._beta()
        weights = (self.size * probs[idx]) ** (-beta)
        weights = weights / weights.max()

        batch = (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx])
        return batch, idx, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """update 후 |TD-error|로 우선순위 갱신."""
        new_p = np.abs(td_errors) + self.epsilon
        self.priorities[indices] = new_p
        self.max_priority = max(self.max_priority, float(new_p.max()))


# ──────────────────────────────────────────────────────────────────────────
# HER (Hindsight Experience Replay, future 전략) + N-step returns
#   실패 에피소드의 각 transition에 대해 "미래에 실제로 도달한 위치"를
#   목표로 재라벨링 → sparse한 성공 신호를 dense하게 만듦.
#   v3: n-step 누적 보상까지 함께 계산해서 push.
# ──────────────────────────────────────────────────────────────────────────

def _push_nstep_trajectory(buffer: PrioritizedReplayBuffer, env: PointCloudNavEnv,
                           episode: list[dict], goal: np.ndarray,
                           n_step: int, gamma: float):
    """v4: 프레임 스택을 반영한 n-step transition push.

    각 raw transition에는 build_obs가 요구하는 goal-무관 스택도 저장돼 있다:
      pre_rays_stack, pre_action_stack:  s_t 관측을 만들 때 사용
      post_rays_stack, post_action_stack: s_{t+1} 관측을 만들 때 사용

    r_bar_t = Σ_{k=0}^{n-1} γ^k · r_{t+k}
    s2_t = s_{end_t+1} (또는 성공 시 s_{end_t}의 next)
    done_t = 구간 내 성공하면 1
    """
    T = len(episode)
    if T == 0:
        return

    # 1) 각 원자 transition의 (r, success) 미리 계산 (충돌 budget은 근사 무시)
    rewards = np.zeros(T, dtype=np.float32)
    successes = np.zeros(T, dtype=bool)
    for t, tr in enumerate(episode):
        r, ok = env.compute_reward(tr["pos"], tr["next_pos"], goal, tr["collided"])
        rewards[t] = r
        successes[t] = ok

    # 2) n-step transition 생성
    for t in range(T):
        r_bar = 0.0
        done = False
        end_t = min(t + n_step - 1, T - 1)
        for k in range(n_step):
            if t + k >= T:
                end_t = T - 1
                break
            r_bar += (gamma ** k) * rewards[t + k]
            if successes[t + k]:
                done = True
                end_t = t + k
                break

        s_t = env.build_obs(episode[t]["pos"], goal,
                            episode[t]["pre_rays_stack"],
                            episode[t]["pre_action_stack"])
        s_next = env.build_obs(episode[end_t]["next_pos"], goal,
                               episode[end_t]["post_rays_stack"],
                               episode[end_t]["post_action_stack"])
        buffer.push(s_t, episode[t]["action"], r_bar, s_next, float(done))


def push_episode_with_her(buffer: PrioritizedReplayBuffer, env: PointCloudNavEnv,
                          episode: list[dict], goal: np.ndarray,
                          k_future: int = 4, n_step: int = 3, gamma: float = 0.99):
    """v4: HER future + n-step. episode 원소는 v4 관측 스택을 포함."""
    T = len(episode)
    if T == 0:
        return
    rng = np.random.default_rng()

    _push_nstep_trajectory(buffer, env, episode, goal, n_step, gamma)

    for _ in range(k_future):
        future_idx = int(rng.integers(0, T))
        her_goal = episode[future_idx]["next_pos"]
        sub_ep = episode[:future_idx + 1]
        _push_nstep_trajectory(buffer, env, sub_ep, her_goal, n_step, gamma)


# ──────────────────────────────────────────────────────────────────────────
# 결과 저장 / GLB / Open3D 시각화
# ──────────────────────────────────────────────────────────────────────────

def default_output_dir() -> Path:
    desktop = Path.home() / "Desktop"
    base = desktop if desktop.exists() else Path.home()
    out = base / "sac_pathway_v4"
    out.mkdir(parents=True, exist_ok=True)
    return out


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


def load_saved_path(path: Path) -> list[np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
    elif suffix == ".json":
        with open(path) as f:
            arr = np.array(json.load(f)["waypoints"])
    else:
        raise ValueError("경로 파일은 .npy, .csv, .json 중 하나여야 합니다.")
    return [np.asarray(w, dtype=float) for w in np.atleast_2d(arr)]


def plot_path_glb(obstacle_points: np.ndarray, path: list[np.ndarray],
                  start: np.ndarray, goal: np.ndarray, output: Path,
                  tube_radius_hint: float = 0.06) -> None:
    """v1과 동일한 GLB 출력 (웹 뷰어/Blender/macOS 미리보기용)."""
    try:
        import trimesh
    except ImportError as e:
        raise ImportError("GLB 시각화를 위해 trimesh가 필요합니다. `pip install trimesh`") from e

    scene = trimesh.Scene()
    if len(obstacle_points) > 0:
        pts = np.asarray(obstacle_points)
        if len(pts) > 80000:
            rng = np.random.default_rng(0)
            pts = pts[rng.choice(len(pts), 80000, replace=False)]
        colors = np.tile(np.array([170, 170, 170, 200], dtype=np.uint8), (len(pts), 1))
        scene.add_geometry(trimesh.PointCloud(pts, colors=colors), geom_name="obstacles")

    path_arr = np.array(path)
    diag = float(np.linalg.norm(path_arr[-1] - path_arr[0])) if len(path_arr) > 1 else 1.0
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
            R = (trimesh.transformations.rotation_matrix(math.pi, [1, 0, 0])
                 if direction[2] < 0 else np.eye(4))
        T = trimesh.transformations.translation_matrix((a + b) / 2)
        cyl.apply_transform(T @ R)
        cyl.visual.face_colors = [255, 140, 0, 255]
        scene.add_geometry(cyl, geom_name=f"path_seg_{i:03d}")

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


def show_open3d(points: np.ndarray, waypoints: list[np.ndarray],
                start: np.ndarray, goal: np.ndarray,
                subgoals: list[np.ndarray] | None = None,
                downsample: int = 5, title: str = "SAC path") -> None:
    """Open3D 인터랙티브 뷰어: 포인트클라우드(높이 컬러맵) + 경로 + 마커.

    조작: 드래그=회전, 스크롤=줌, Shift+드래그=이동, Q=닫기
    """
    try:
        import open3d as o3d
    except ImportError as e:
        raise ImportError("Open3D가 필요합니다: pip install open3d") from e

    geometries = []

    # 1) 포인트클라우드 — 높이(z)에 따라 색을 입혀 층 구분이 보이게
    pts = np.asarray(points, dtype=float)[:: max(1, downsample)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    z = pts[:, 2]
    z_norm = (z - z.min()) / max(z.max() - z.min(), 1e-9)
    # 낮은 층 = 어두운 회청색, 높은 층 = 밝은 베이지 (경로 색과 겹치지 않게)
    colors = np.stack([0.45 + 0.35 * z_norm,
                       0.45 + 0.30 * z_norm,
                       0.55 + 0.15 * z_norm], axis=1)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(pcd)

    # 2) 경로 — 주황색 LineSet + waypoint 구
    wp = np.array(waypoints, dtype=float)
    if len(wp) >= 2:
        lines = [[i, i + 1] for i in range(len(wp) - 1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(wp)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([[1.0, 0.55, 0.0]] * len(lines))
        geometries.append(ls)
    for p in wp[1:-1]:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sph.translate(p)
        sph.paint_uniform_color([1.0, 0.55, 0.0])
        geometries.append(sph)

    # 3) subgoal (door) — 노란색 구
    if subgoals:
        for sg in subgoals[:-1]:  # 마지막은 goal이므로 제외
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
            sph.translate(np.asarray(sg, dtype=float))
            sph.paint_uniform_color([1.0, 0.85, 0.1])
            geometries.append(sph)

    # 4) 시작(초록) / 목표(빨강)
    s_sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.18)
    s_sph.translate(np.asarray(start, dtype=float))
    s_sph.paint_uniform_color([0.15, 0.8, 0.3])
    geometries.append(s_sph)

    g_sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.18)
    g_sph.translate(np.asarray(goal, dtype=float))
    g_sph.paint_uniform_color([0.9, 0.2, 0.2])
    geometries.append(g_sph)

    print("[Open3D] 드래그=회전, 스크롤=줌, Shift+드래그=이동, Q=닫기")
    o3d.visualization.draw_geometries(geometries, window_name=title,
                                      width=1280, height=800)


# ──────────────────────────────────────────────────────────────────────────
# 학습 루프 (SAC v2 + HER + 커리큘럼)
# ──────────────────────────────────────────────────────────────────────────

def train(args) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"입력 로드 중: {args.input}")
    points = load_points(Path(args.input))
    print(f"  포인트 수: {len(points):,}")

    gm = voxelize(points, args.resolution, args.margin, args.sample)
    print(f"격자 크기: {gm.shape}, 자유 공간 비율: {gm.free_ratio * 100:.1f}%")

    env = PointCloudNavEnv(
        gm, max_episode_steps=args.max_episode_steps, seed=args.seed,
        ray_range=args.ray_range, ray_density=args.ray_density,
        frame_stack=args.frame_stack,
        max_consec_collisions=args.max_consec_collisions,
        collision_cap=args.collision_cap,
    )
    fixed = args.fixed_start is not None and args.fixed_goal is not None
    fixed_start = np.asarray(args.fixed_start, dtype=float) if fixed else None
    fixed_goal = np.asarray(args.fixed_goal, dtype=float) if fixed else None

    use_her = not args.no_her
    use_curriculum = (not args.no_curriculum) and not fixed
    use_per = not args.no_per
    n_step = max(1, args.n_step)
    print(f"학습 모드: {'고정 시작/목표 쌍' if fixed else 'goal-conditioned'}"
          f" | SAC v4(LayerNorm + entropy anneal) | HER={'ON(k=' + str(args.her_k) + ')' if use_her else 'OFF'}"
          f" | PER={'ON' if use_per else 'OFF'} | n-step={n_step}"
          f" | 커리큘럼={'adaptive' if use_curriculum else 'OFF'}"
          f" | rays={env.n_rays}방향×{env.frame_stack}프레임")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    agent = SACv4Agent(env.OBS_DIM, env.ACT_DIM, gamma=args.gamma, tau=args.tau, lr=args.lr)
    # v4: target entropy annealing 설정
    agent.set_target_entropy_schedule(args.target_entropy_end, env.ACT_DIM)
    entropy_anneal_steps = int(args.total_steps * args.target_entropy_frac)
    print(f"디바이스: {agent.device}")
    print(f"  target_entropy: {agent.target_entropy_start:.1f} → {agent.target_entropy_end:.1f} "
          f"(첫 {entropy_anneal_steps:,} 스텝에 걸쳐)")
    buffer = PrioritizedReplayBuffer(
        env.OBS_DIM, env.ACT_DIM, capacity=args.buffer_size,
        alpha=args.per_alpha if use_per else 0.0,  # α=0이면 uniform
        beta_frames=args.total_steps,
    )

    log_path = out_dir / "training_log.csv"
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["episode", "env_steps", "return", "length", "success",
                         "collisions", "final_dist_m", "goal_dist_m", "alpha",
                         "curriculum_max_dist_m", "target_entropy", "stuck"])

    # ── Adaptive curriculum ──
    # 성공률이 expand_threshold를 넘으면 확장, shrink_threshold 아래면 축소.
    # 확장/축소는 rate만큼 승수로 (곱연산).
    cur_max = args.curriculum_start if use_curriculum else None
    expand_th, shrink_th = 0.6, 0.2
    expand_rate, shrink_rate = 1.08, 0.95  # 확장은 8%씩, 축소는 5%씩

    def clamp_curriculum(v: float) -> float:
        return float(np.clip(v, args.curriculum_start, env.scene_diag))

    obs = env.reset(fixed_start, fixed_goal, max_goal_dist=cur_max)
    ep_goal_dist = float(np.linalg.norm(env.goal - env.pos))
    ep_transitions: list[dict] = []
    ep_return, ep_len, ep_collisions, episode = 0.0, 0, 0, 0
    recent_success: list[int] = []
    best_success_rate = -1.0
    last_alpha = 1.0
    t0 = time.perf_counter()

    # ── Best-model rollback ──
    # best 스냅샷을 메모리에 유지. best 대비 collapse_drop 이상 급락하면 복원.
    best_snapshot: dict | None = None
    rollback_count = 0

    config = {"version": "v4", "resolution": args.resolution, "margin": args.margin,
              "sample": args.sample, "max_step": env.max_step,
              "goal_threshold": env.goal_threshold, "ray_range": env.ray_range,
              "ray_density": args.ray_density, "frame_stack": args.frame_stack,
              "max_consec_collisions": args.max_consec_collisions,
              "collision_cap": args.collision_cap,
              "max_episode_steps": args.max_episode_steps,
              "obs_dim": env.OBS_DIM, "act_dim": env.ACT_DIM,
              "her": use_her, "her_k": args.her_k, "n_step": n_step,
              "per": use_per, "curriculum": use_curriculum,
              "target_entropy_end": args.target_entropy_end,
              "target_entropy_frac": args.target_entropy_frac,
              "input": str(args.input)}

    ep_stuck_count = 0
    for step in range(1, args.total_steps + 1):
        # v4: target entropy annealing (매 스텝마다)
        agent.update_target_entropy(step / entropy_anneal_steps)

        if step <= args.warmup_steps:
            action = np.random.uniform(-1, 1, size=env.ACT_DIM).astype(np.float32)
        else:
            action = agent.act(obs)

        # v4: HER용 스택 스냅샷 (step 이전 상태)
        pre_rays_stack = env.rays_stack.copy()
        pre_action_stack = env.action_stack.copy()
        prev_pos = env.pos.copy()

        next_obs, reward, done, info = env.step(action)

        if use_her:
            ep_transitions.append({
                "pos": prev_pos,
                "action": np.asarray(action, dtype=np.float32),
                "next_pos": env.pos.copy(),
                "collided": info["collided"],
                # v4: 프레임 스택 스냅샷 (step 전/후)
                "pre_rays_stack": pre_rays_stack,
                "pre_action_stack": pre_action_stack,
                "post_rays_stack": env.rays_stack.copy(),
                "post_action_stack": env.action_stack.copy(),
            })
        else:
            # HER OFF: 단일 step transition을 그대로 push (n-step 미적용)
            buffer.push(obs, action, reward, next_obs, float(done and info["success"]))

        obs = next_obs
        ep_return += reward
        ep_len += 1
        ep_collisions += int(info["collided"])

        # ── SAC update with PER (IS weights + priority 갱신) ──
        if step > args.warmup_steps and buffer.size >= args.batch_size:
            for _ in range(args.gradient_steps):
                batch, idx, is_w = buffer.sample(args.batch_size)
                stats = agent.update(batch, is_weights=is_w,
                                     n_step=n_step if use_her else 1)
                buffer.update_priorities(idx, stats["td_error"])
                last_alpha = stats["alpha"]

        if done:
            episode += 1
            if info.get("stuck"):
                ep_stuck_count += 1
            if use_her:
                push_episode_with_her(buffer, env, ep_transitions, env.goal.copy(),
                                      k_future=args.her_k, n_step=n_step,
                                      gamma=args.gamma)
                ep_transitions = []
            recent_success.append(int(info["success"]))
            if len(recent_success) > 50:
                recent_success.pop(0)
            log_writer.writerow([episode, step, round(ep_return, 2), ep_len,
                                 int(info["success"]), ep_collisions,
                                 round(info["dist"], 3), round(ep_goal_dist, 3),
                                 round(last_alpha, 4),
                                 round(cur_max, 2) if cur_max else -1,
                                 round(agent.target_entropy, 3),
                                 int(info.get("stuck", False))])
            if episode % args.log_every == 0:
                sr = float(np.mean(recent_success))
                elapsed = time.perf_counter() - t0
                cd = f"{cur_max:.1f}m" if cur_max else "∞"
                stuck_frac = ep_stuck_count / episode * 100
                print(f"  ep {episode:5d} | step {step:8,d} | return {ep_return:8.2f} | "
                      f"len {ep_len:4d} | 최근50 성공률 {sr * 100:5.1f}% | "
                      f"α {last_alpha:.3f} | H_t {agent.target_entropy:.2f} | "
                      f"커리큘럼 {cd} | stuck {stuck_frac:.1f}% | {elapsed:7.1f}s")
                log_file.flush()

                # Best 갱신 + 스냅샷 저장
                if sr >= best_success_rate:
                    best_success_rate = sr
                    agent.save(out_dir / "sac_model_best.pt", config)
                    if sr >= 0.3:  # 최소 30% 이상일 때만 rollback 대상
                        best_snapshot = agent.state_snapshot()

                # ── Catastrophic collapse 감지 → best로 롤백 ──
                # best가 30% 이상이고, 최근 성공률이 best-collapse_drop 이하면 롤백
                if (best_snapshot is not None and best_success_rate >= 0.3
                        and sr <= best_success_rate - args.collapse_drop
                        and len(recent_success) >= 50):
                    rollback_count += 1
                    print(f"  ⚠ [Rollback #{rollback_count}] 성공률 {sr * 100:.1f}% "
                          f"< best {best_success_rate * 100:.1f}%-{args.collapse_drop * 100:.0f}%p → "
                          f"best 모델로 되감기")
                    agent.load_snapshot(best_snapshot)
                    recent_success = []  # 통계 리셋 (best에서 다시 시작)

                # ── Adaptive curriculum: 성공률 기반으로 목표 거리 조절 ──
                if use_curriculum and len(recent_success) >= 20:
                    if sr >= expand_th and cur_max < env.scene_diag:
                        cur_max = clamp_curriculum(cur_max * expand_rate)
                    elif sr <= shrink_th and cur_max > args.curriculum_start:
                        cur_max = clamp_curriculum(cur_max * shrink_rate)

            obs = env.reset(fixed_start, fixed_goal, max_goal_dist=cur_max)
            ep_goal_dist = float(np.linalg.norm(env.goal - env.pos))
            ep_return, ep_len, ep_collisions = 0.0, 0, 0

        if step % args.checkpoint_every == 0:
            agent.save(out_dir / "sac_model.pt", config)

    agent.save(out_dir / "sac_model.pt", config)
    log_file.close()
    print(f"\n[완료] 총 {args.total_steps:,} env steps 학습 "
          f"(best 성공률 {best_success_rate * 100:.1f}%, rollback {rollback_count}회)")
    print(f"  모델 저장:      {(out_dir / 'sac_model.pt').resolve()}")
    print(f"  best 모델 저장: {(out_dir / 'sac_model_best.pt').resolve()}")
    print(f"  학습 로그:      {log_path.resolve()}")


# ──────────────────────────────────────────────────────────────────────────
# 경로 탐색 (계층적 subgoal rollout)
# ──────────────────────────────────────────────────────────────────────────

def _rollout_to(env: PointCloudNavEnv, agent: SACv4Agent, start: np.ndarray,
                subgoal: np.ndarray, threshold: float, max_steps: int,
                deterministic: bool) -> tuple[list[np.ndarray], bool]:
    """현재 위치 start에서 subgoal까지 정책 rollout. 반환: (waypoints, 도달 여부)"""
    saved_thr = env.goal_threshold
    env.goal_threshold = threshold
    obs = env.reset(start, subgoal)
    wps = [env.pos.copy()]
    ok = False
    for _ in range(max_steps):
        action = agent.act(obs, deterministic=deterministic)
        obs, _, done, info = env.step(action)
        wps.append(env.pos.copy())
        if done:
            ok = info["success"]
            break
    env.goal_threshold = saved_thr
    return wps, ok


def plan(args) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model) if args.model else out_dir / "sac_model_best.pt"
    if not model_path.exists():
        fallback = out_dir / "sac_model.pt"
        if fallback.exists():
            model_path = fallback
        else:
            raise SystemExit(f"[에러] 모델 파일을 찾을 수 없습니다: {model_path}")

    # v4: 먼저 config만 읽어 env를 학습 시 설정 그대로 재현
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})

    print(f"입력 로드 중: {args.input}")
    points = load_points(Path(args.input))
    print(f"  포인트 수: {len(points):,}")

    resolution = config.get("resolution", args.resolution)
    margin = config.get("margin", args.margin)
    sample = config.get("sample", args.sample)
    gm = voxelize(points, resolution, margin, sample)
    print(f"격자 크기: {gm.shape}, 자유 공간 비율: {gm.free_ratio * 100:.1f}%")

    env = PointCloudNavEnv(
        gm, max_step=config.get("max_step"),
        max_episode_steps=args.max_episode_steps,
        goal_threshold=config.get("goal_threshold"),
        ray_range=config.get("ray_range", 5.0),
        ray_density=config.get("ray_density", 1),
        frame_stack=config.get("frame_stack", 3),
        max_consec_collisions=config.get("max_consec_collisions", 5),
        collision_cap=config.get("collision_cap", 30.0),
    )

    # 이제 env의 obs_dim을 알았으니 agent 로드
    agent, _ = SACv4Agent.load(model_path, env.OBS_DIM, env.ACT_DIM)
    print(f"모델 로드: {model_path}")
    print(f"  관측 차원: {env.OBS_DIM} "
          f"(rays {env.n_rays}방향×{env.frame_stack}프레임)")

    start = snap_to_free(gm, np.asarray(args.start, dtype=float))
    goal = snap_to_free(gm, np.asarray(args.goal, dtype=float))
    print(f"시작점: {tuple(round(float(v), 2) for v in start)}  "
          f"목표점: {tuple(round(float(v), 2) for v in goal)} (자유 공간 스냅 적용)")

    # ── subgoal 생성 (rooms_graph.json 있으면 계층적, 없으면 직접) ──
    subgoals = [goal]
    rooms_traversed = None
    if args.rooms_json is not None:
        if not args.rooms_json.exists():
            raise SystemExit(f"[에러] rooms-json 파일 없음: {args.rooms_json}")
        subgoals, rooms_traversed = make_subgoals(start, goal, args.rooms_json, gm)
        print(f"  subgoal 수: {len(subgoals)} (door {len(subgoals) - 1}개 + 최종 목표)")

    # ── 세그먼트별 rollout: deterministic 1회 → 실패 시 stochastic 재시도 ──
    door_threshold = max(gm.resolution * 3.0, env.goal_threshold)
    t0 = time.perf_counter()
    waypoints: list[np.ndarray] = [start.copy()]
    cur = start.copy()
    success = True
    for i, sg in enumerate(subgoals):
        is_final = (i == len(subgoals) - 1)
        thr = env.goal_threshold if is_final else door_threshold
        seg, ok = _rollout_to(env, agent, cur, sg, thr, args.max_episode_steps, True)
        if not ok:
            for _ in range(args.retries):
                seg2, ok = _rollout_to(env, agent, cur, sg, thr,
                                       args.max_episode_steps, False)
                if ok:
                    seg = seg2
                    break
        label = "최종 목표" if is_final else f"subgoal {i + 1} (door)"
        print(f"  [{i + 1}/{len(subgoals)}] {label}: "
              f"{'도달' if ok else '미도달'} ({len(seg)} wps)")
        waypoints.extend(seg[1:])
        cur = waypoints[-1].copy()
        if not ok:
            success = False
            break
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if not args.no_smooth:
        waypoints = smooth_path(waypoints, gm)

    residual = float(np.linalg.norm(goal - waypoints[-1]))
    print(f"\n[{'성공' if success else '주의'}] SAC v4 경로 rollout "
          f"{'성공' if success else '실패 (일부 구간 미도달)'}")
    print(f"  소요 시간:    {elapsed_ms:.1f} ms")
    print(f"  경로 길이:    {path_length(waypoints):.2f} m")
    print(f"  Waypoint 수:  {len(waypoints)}")
    print(f"  목표까지 잔여 거리: {residual:.3f} m")
    if rooms_traversed:
        print(f"  방 시퀀스:    {' → '.join(rooms_traversed)}")

    if args.save:
        save_target = Path(args.save)
        if not save_target.is_absolute():
            save_target = out_dir / save_target
    else:
        save_target = out_dir / "sac_path.json"
    save_path(waypoints, save_target)
    print(f"  경로 저장:    {save_target.resolve()}")

    result_meta = {
        "success": bool(success),
        "start": start.tolist(), "goal": goal.tolist(),
        "distance_m": path_length(waypoints), "waypoints": len(waypoints),
        "residual_dist_m": residual, "rooms_traversed": rooms_traversed,
        "model": str(model_path),
    }
    with open(out_dir / "sac_path_meta.json", "w") as f:
        json.dump(result_meta, f, indent=2, ensure_ascii=False)

    if args.plot:
        occ = np.argwhere(gm.grid == 1)
        obstacles = (occ * gm.resolution + np.array([gm.x_min, gm.y_min, gm.z_min])
                     if len(occ) > 0 else np.empty((0, 3)))
        plot_target = out_dir / args.plot_output
        plot_path_glb(obstacles, waypoints, start, goal, plot_target,
                      tube_radius_hint=max(gm.resolution * 0.4, 0.05))
        print(f"  GLB 저장:     {plot_target.resolve()}")

    if args.show:
        try:
            show_open3d(points, waypoints, start, goal,
                        subgoals=subgoals if len(subgoals) > 1 else None,
                        downsample=args.show_downsample,
                        title=f"SAC v4 path ({'success' if success else 'fail'})")
        except ImportError as e:
            print(f"\n  [알림] Open3D 시각화 건너뜀 — {e}")
            print(f"  → pip install open3d 후 'show' 명령으로 다시 볼 수 있어요:")
            print(f"    python {Path(__file__).name} show --input {args.input} "
                  f"--path {save_target}")


# ──────────────────────────────────────────────────────────────────────────
# show: 저장된 경로를 Open3D로 보기
# ──────────────────────────────────────────────────────────────────────────

def show(args) -> None:
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    path_file = Path(args.path) if args.path else out_dir / "sac_path.json"
    if not path_file.exists():
        raise SystemExit(f"[에러] 경로 파일을 찾을 수 없습니다: {path_file}")

    print(f"입력 로드 중: {args.input}")
    points = load_points(Path(args.input))
    print(f"  포인트 수: {len(points):,}")

    waypoints = load_saved_path(path_file)
    print(f"경로 로드: {path_file} ({len(waypoints)} waypoints)")

    meta_file = out_dir / "sac_path_meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        start = np.array(meta["start"]); goal = np.array(meta["goal"])
    else:
        start, goal = waypoints[0], waypoints[-1]

    try:
        show_open3d(points, waypoints, start, goal,
                    downsample=args.show_downsample, title=path_file.name)
    except ImportError as e:
        raise SystemExit(f"[에러] {e}")


# ──────────────────────────────────────────────────────────────────────────
# curve: learning curve (v1과 동일 + α 패널 추가)
# ──────────────────────────────────────────────────────────────────────────

def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def curve(args) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib가 필요합니다: pip install matplotlib") from e

    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    log_path = Path(args.log) if args.log else out_dir / "training_log.csv"
    if not log_path.exists():
        raise SystemExit(f"[에러] 학습 로그를 찾을 수 없습니다: {log_path}")

    episodes, env_steps, returns, lengths, successes, alphas = [], [], [], [], [], []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            env_steps.append(int(row["env_steps"]))
            returns.append(float(row["return"]))
            lengths.append(int(row["length"]))
            successes.append(int(row["success"]))
            alphas.append(float(row.get("alpha", 0) or 0))

    if not episodes:
        raise SystemExit(f"[에러] 로그에 데이터가 없습니다: {log_path}")

    episodes = np.array(episodes); env_steps = np.array(env_steps)
    returns = np.array(returns, dtype=float); lengths = np.array(lengths, dtype=float)
    successes = np.array(successes, dtype=float); alphas = np.array(alphas, dtype=float)
    x = env_steps if args.x_axis == "steps" else episodes
    x_label = "env steps" if args.x_axis == "steps" else "episode"
    w = args.smooth
    sw = args.success_window

    print(f"로그 로드: {log_path}  (총 {len(episodes):,} 에피소드)")
    print(f"  최종 최근{sw} 성공률: {np.mean(successes[-sw:]) * 100:.1f}%")

    has_alpha = bool(np.any(alphas > 0))
    n_panels = 4 if has_alpha else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 3.6 * n_panels), sharex=True)

    axes[0].plot(x, returns, color="#cccccc", linewidth=0.8, label="raw")
    if len(returns) >= w > 1:
        axes[0].plot(x[w - 1:], _moving_average(returns, w), color="#ff8c00",
                     linewidth=2.0, label=f"MA(w={w})")
    axes[0].set_ylabel("return"); axes[0].set_title("SAC v4 Learning Curve — Return")
    axes[0].legend(loc="best"); axes[0].grid(True, alpha=0.3)

    rolling_sr = np.array([np.mean(successes[max(0, i - sw + 1):i + 1]) * 100
                           for i in range(len(successes))])
    axes[1].plot(x, rolling_sr, color="#2e8b57", linewidth=2.0)
    axes[1].set_ylabel("success rate (%)")
    axes[1].set_title(f"Rolling success (window={sw})")
    axes[1].set_ylim(-2, 102); axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, lengths, color="#cccccc", linewidth=0.8, label="raw")
    if len(lengths) >= w > 1:
        axes[2].plot(x[w - 1:], _moving_average(lengths, w), color="#4169e1",
                     linewidth=2.0, label=f"MA(w={w})")
    axes[2].set_ylabel("episode length"); axes[2].set_title("Episode length")
    axes[2].legend(loc="best"); axes[2].grid(True, alpha=0.3)

    if has_alpha:
        axes[3].plot(x, alphas, color="#8b3a9e", linewidth=1.5)
        axes[3].set_ylabel("alpha")
        axes[3].set_title("Entropy temperature α (auto-tuned)")
        axes[3].grid(True, alpha=0.3)

    axes[-1].set_xlabel(x_label)
    plt.tight_layout()
    out_path = out_dir / args.curve_output
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"  learning curve 저장: {out_path.resolve()}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input", type=Path, required=True,
                   help="포인트클라우드 입력: .npy/.ply 파일, 방 폴더, 또는 상위 폴더(npy/)")
    p.add_argument("--resolution", type=float, default=0.15, help="voxel 한 칸의 크기 (m)")
    p.add_argument("--margin", type=int, default=0, help="장애물 팽창 마진 (voxel)")
    p.add_argument("--sample", type=int, default=10, help="포인트클라우드 다운샘플링 간격")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="결과 저장 폴더 (기본: ~/Desktop/sac_pathway_v2)")
    p.add_argument("--max-episode-steps", type=int, default=500, help="세그먼트당 최대 스텝")
    p.add_argument("--seed", type=int, default=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="개선된 SAC(v2+HER+커리큘럼+계층적 planning)로 3D 경로를 학습/탐색합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    pt = sub.add_parser("train", help="SAC v4 정책 학습 (v3 + 관측/환경 layer 개선)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(pt)
    pt.add_argument("--total-steps", type=int, default=300_000, help="총 환경 스텝 수")
    pt.add_argument("--warmup-steps", type=int, default=2_000, help="랜덤 행동 warmup")
    pt.add_argument("--batch-size", type=int, default=256)
    pt.add_argument("--buffer-size", type=int, default=1_000_000,
                    help="replay buffer 크기 (HER로 transition이 ~5배 늘어남)")
    pt.add_argument("--gradient-steps", type=int, default=1)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--gamma", type=float, default=0.99)
    pt.add_argument("--tau", type=float, default=0.005)
    pt.add_argument("--no-her", action="store_true", help="HER 끄기")
    pt.add_argument("--her-k", type=int, default=4, help="transition당 HER 재라벨 수")
    pt.add_argument("--n-step", type=int, default=3,
                    help="[v3] n-step return의 n (1이면 원래 1-step SAC)")
    pt.add_argument("--no-per", action="store_true",
                    help="[v3] Prioritized Experience Replay 끄기 (uniform 샘플링으로)")
    pt.add_argument("--per-alpha", type=float, default=0.6,
                    help="[v3] PER 우선순위 지수 α (0=uniform, 1=full)")
    pt.add_argument("--collapse-drop", type=float, default=0.25,
                    help="[v3] best 대비 이만큼(단위: 비율) 급락하면 best 모델로 롤백")
    pt.add_argument("--no-curriculum", action="store_true", help="커리큘럼 끄기")
    pt.add_argument("--curriculum-start", type=float, default=3.0,
                    help="[v3] adaptive 커리큘럼 초기(및 최소) 목표 거리 (m)")
    pt.add_argument("--curriculum-steps", type=int, default=150_000,
                    help="(v2 호환용, v3+ adaptive에서는 사용 안 함)")
    # ── v4: 환경/관측 layer ──
    pt.add_argument("--ray-range", type=float, default=5.0,
                    help="[v4] raycast 최대 거리 (m). v3은 3.0")
    pt.add_argument("--ray-density", type=int, default=1, choices=[1, 2],
                    help="[v4] 1=26방향(3x3x3), 2=~98방향(5x5x5 primitive)")
    pt.add_argument("--frame-stack", type=int, default=3,
                    help="[v4] 관측에 쌓을 프레임 수 (rays + 이전 행동)")
    pt.add_argument("--max-consec-collisions", type=int, default=5,
                    help="[v4] 연속 이만큼 충돌하면 에피소드 조기 종료")
    pt.add_argument("--collision-cap", type=float, default=30.0,
                    help="[v4] 에피소드당 총 충돌 페널티 상한")
    pt.add_argument("--target-entropy-end", type=float, default=-6.0,
                    help="[v4] annealing 종료 시점 target_entropy (초기값은 -|A|=-3)")
    pt.add_argument("--target-entropy-frac", type=float, default=0.7,
                    help="[v4] 총 스텝 중 몇 %를 target_entropy anneal에 쓸지 (0~1)")
    pt.add_argument("--fixed-start", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    pt.add_argument("--fixed-goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    pt.add_argument("--log-every", type=int, default=10)
    pt.add_argument("--checkpoint-every", type=int, default=10_000)
    pt.set_defaults(func=train)

    # ── plan ──
    pp = sub.add_parser("plan", help="학습된 정책으로 경로 탐색 (rooms-json 시 계층적)",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_common(pp)
    pp.add_argument("--model", type=Path, default=None,
                    help="모델 체크포인트 (기본: <output-dir>/sac_model_best.pt)")
    pp.add_argument("--rooms-json", type=Path, default=None,
                    help="rooms_graph.json — 지정 시 door subgoal 기반 계층적 planning")
    pp.add_argument("--start", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    pp.add_argument("--goal", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    pp.add_argument("--save", type=Path, default=None,
                    help="경로 저장 파일명 (.npy/.csv/.json, 기본: sac_path.json)")
    pp.add_argument("--retries", type=int, default=10,
                    help="세그먼트별 stochastic 재시도 횟수")
    pp.add_argument("--no-smooth", action="store_true")
    pp.add_argument("--plot", action="store_true", help="GLB 파일 저장")
    pp.add_argument("--plot-output", type=Path, default=Path("sac_path.glb"))
    pp.add_argument("--show", action="store_true", help="Open3D 인터랙티브 뷰어 실행")
    pp.add_argument("--show-downsample", type=int, default=5,
                    help="Open3D 포인트 다운샘플링 간격")
    pp.set_defaults(func=plan)

    # ── show ──
    ps = sub.add_parser("show", help="저장된 경로를 Open3D로 보기",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ps.add_argument("--input", type=Path, required=True,
                    help="포인트클라우드 입력 (npy 폴더 등)")
    ps.add_argument("--path", type=Path, default=None,
                    help="경로 파일 (기본: <output-dir>/sac_path.json)")
    ps.add_argument("--output-dir", type=Path, default=None)
    ps.add_argument("--show-downsample", type=int, default=5)
    ps.set_defaults(func=show)

    # ── curve ──
    pc = sub.add_parser("curve", help="training_log.csv로 learning curve 그리기",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pc.add_argument("--log", type=Path, default=None)
    pc.add_argument("--output-dir", type=Path, default=None)
    pc.add_argument("--curve-output", type=Path, default=Path("learning_curve.png"))
    pc.add_argument("--x-axis", choices=["episode", "steps"], default="steps")
    pc.add_argument("--smooth", type=int, default=20)
    pc.add_argument("--success-window", type=int, default=50)
    pc.add_argument("--dpi", type=int, default=130)
    pc.set_defaults(func=curve)

    args = parser.parse_args()
    if getattr(args, "input", None) is not None and not args.input.exists():
        raise SystemExit(f"[에러] 입력을 찾을 수 없습니다: {args.input}")
    args.func(args)


if __name__ == "__main__":
    main()
