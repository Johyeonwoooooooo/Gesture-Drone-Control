#!/usr/bin/env python3
"""건축 평면도 + 실제 A* 경로 시각화.
rooms_graph.json + 3D.py를 사용해서 각 방에 대해 실제 A*를 실행하고
세세한 waypoint까지 모두 시각화합니다."""

from __future__ import annotations

import json
import sys
import platform
import importlib.util
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle


# ──────────────────────────────────────────────────────────────────────────────
# 한글 폰트 자동 설정
# ──────────────────────────────────────────────────────────────────────────────

def setup_korean_font():
    """OS별 한글 폰트 자동 선택."""
    candidates = [
        'AppleGothic',  # macOS
        'AppleSDGothicNeo',  # macOS
        'NanumGothic',  # Linux
        'Malgun Gothic',  # Windows
        'Noto Sans CJK KR',  # Linux/노트북
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✓ 한글 폰트: {font}")
            return font
    print("⚠ 한글 폰트 없음 — 기본 폰트 사용 (한글이 깨질 수 있음)")
    return None


setup_korean_font()

# ──────────────────────────────────────────────────────────────────────────────
# 3D.py 동적 import
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
PATHFINDER_PATH = SCRIPT_DIR / "3D.py"
PATHFINDER_AVAILABLE = False

if PATHFINDER_PATH.exists():
    try:
        spec = importlib.util.spec_from_file_location("pathfinder", PATHFINDER_PATH)
        pathfinder = importlib.util.module_from_spec(spec)
        sys.modules["pathfinder"] = pathfinder
        spec.loader.exec_module(pathfinder)
        voxelize = pathfinder.voxelize
        astar = pathfinder.astar
        smooth_path = pathfinder.smooth_path
        PATHFINDER_AVAILABLE = True
        print(f"✓ 3D.py 로드 성공")
    except Exception as e:
        print(f"⚠ 3D.py 로드 실패: {e}")
else:
    print(f"⚠ 3D.py 없음: A* 계산 불가, 직선만 표시")


# ──────────────────────────────────────────────────────────────────────────────
# 유틸: JSON / 그래프 / 좌표
# ──────────────────────────────────────────────────────────────────────────────

def load_rooms_graph(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def find_door_between(room_a, room_b, edges):
    """두 방 사이의 문 위치와 타입 찾기."""
    for edge in edges:
        if (edge["a"] == room_a and edge["b"] == room_b) or \
                (edge["a"] == room_b and edge["b"] == room_a):
            return np.array(edge["door_center"], dtype=float), edge.get("type", "door")
    return None, None


def find_path_between_rooms(start_room_id, goal_room_id, graph_data):
    """BFS로 방 시퀀스 찾기."""
    edges = graph_data.get("edges", [])
    adj = {}
    for edge in edges:
        a, b = edge["a"], edge["b"]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    queue = deque([(start_room_id, [start_room_id])])
    visited = {start_room_id}
    while queue:
        current, path = queue.popleft()
        if current == goal_room_id:
            return path
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    return [start_room_id]


def find_room_by_point(point, rooms):
    """좌표 → 방 ID."""
    p = np.array(point)
    for room_id, room_info in rooms.items():
        bbox_min = np.array(room_info["bbox_min"])
        bbox_max = np.array(room_info["bbox_max"])
        if np.all(bbox_min <= p) and np.all(p <= bbox_max):
            return room_id
    return None


def project_inside_bbox(pt, bbox_min, bbox_max, margin=0.3):
    """점을 bbox 안쪽으로 살짝 클립 (벽 위에 있는 문을 방 안으로 끌어옴)."""
    return np.clip(np.array(pt, dtype=float), bbox_min + margin, bbox_max - margin)


# ──────────────────────────────────────────────────────────────────────────────
# 실제 A* 경로 계산 (방별)
# ──────────────────────────────────────────────────────────────────────────────

def compute_actual_paths(graph_data, path_rooms, global_start, global_goal,
                         npy_dir, resolution=0.15, sample=10, margin=0):
    """경로상의 각 방에 대해 A*를 실행하여 실제 waypoint 추출.

    각 방의 entry/exit:
      - 첫 방: entry=global_start, exit=다음방으로의 문
      - 중간 방: entry=이전방에서의 문, exit=다음방으로의 문
      - 마지막 방: entry=이전방에서의 문, exit=global_goal
    """
    if not PATHFINDER_AVAILABLE:
        return None

    rooms = graph_data["rooms"]
    edges = graph_data["edges"]
    segments = []

    for i, room_id in enumerate(path_rooms):
        if room_id not in rooms:
            continue

        room_info = rooms[room_id]
        bbox_min = np.array(room_info["bbox_min"])
        bbox_max = np.array(room_info["bbox_max"])

        # entry 결정
        if i == 0:
            entry_orig = np.array(global_start, dtype=float)
        else:
            door, _ = find_door_between(path_rooms[i - 1], room_id, edges)
            entry_orig = door if door is not None else np.array(room_info["center"])

        # exit 결정
        if i == len(path_rooms) - 1:
            exit_orig = np.array(global_goal, dtype=float)
        else:
            door, _ = find_door_between(room_id, path_rooms[i + 1], edges)
            exit_orig = door if door is not None else np.array(room_info["center"])

        # bbox 안쪽으로 클립 (A* 가능하도록)
        entry = project_inside_bbox(entry_orig, bbox_min, bbox_max)
        exit_pt = project_inside_bbox(exit_orig, bbox_min, bbox_max)

        # A* 실행
        npy_path = Path(npy_dir) / room_info["npy"] / "coord.npy"

        seg = {
            "room": room_id,
            "floor": room_info["floor"],
            "entry_orig": entry_orig,
            "exit_orig": exit_orig,
            "waypoints": [entry_orig, exit_orig],
            "method": "직선",
        }

        if not npy_path.exists():
            print(f"  ⚠ {room_id}: {npy_path.name} 없음 → 직선")
            segments.append(seg)
            continue

        try:
            points = np.load(npy_path).astype(float)
            if points.ndim == 1:
                points = points.reshape(-1, 3)
            points = points[:, :3]

            gm = voxelize(points, resolution, margin, sample)
            a_path, _ = astar(gm, entry, exit_pt)

            if a_path is not None and len(a_path) > 0:
                a_path = smooth_path(a_path, gm)
                # 원래 entry(이전방 문), exit(다음방 문) 양 끝에 붙임
                waypoints = ([entry_orig] +
                             [np.array(p) for p in a_path] +
                             [exit_orig])
                seg["waypoints"] = waypoints
                seg["method"] = "A*"
                print(f"  ✓ {room_id} (F{room_info['floor']}): A* {len(a_path)} waypoints")
            else:
                print(f"  ⚠ {room_id}: A* 실패 → 직선")
        except Exception as e:
            print(f"  ⚠ {room_id}: {e}")

        segments.append(seg)

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────

# 색상 팔레트
COLOR_START_FILL = "#a8e6a3"
COLOR_START_EDGE = "#2d8c2d"
COLOR_GOAL_FILL = "#f7a8a8"
COLOR_GOAL_EDGE = "#cc2929"
COLOR_PATH_FILL = "#ffd089"
COLOR_PATH_EDGE = "#e67e00"
COLOR_OTHER_FILL = "#e8e8e8"
COLOR_OTHER_EDGE = "#888888"
COLOR_PATH_LINE = "#e91e63"
COLOR_DOOR_PATH = "#d32f2f"
COLOR_DOOR_OTHER = "#aaaaaa"


def visualize(json_path, global_start, global_goal, output_path,
              npy_dir=None, resolution=0.15, sample=10, margin=0):
    """전체 평면도 + 실제 A* 경로 시각화."""

    graph_data = load_rooms_graph(json_path)
    rooms = graph_data["rooms"]
    edges = graph_data["edges"]

    # Floor별 정리
    floors = {}
    for room_id, room_info in rooms.items():
        floors.setdefault(room_info["floor"], []).append((room_id, room_info))

    # 시작/목표 방 찾기 (bbox 안에 없으면 가장 가까운 방)
    start_room_id = find_room_by_point(global_start, rooms)
    goal_room_id = find_room_by_point(global_goal, rooms)

    if start_room_id is None:
        start_room_id = min(rooms.items(),
                            key=lambda x: np.linalg.norm(
                                np.array(global_start[:2]) - np.array(x[1]["center"][:2])))[0]
        print(f"⚠ 시작점 → 가장 가까운 방: {start_room_id}")

    if goal_room_id is None:
        goal_room_id = min(rooms.items(),
                           key=lambda x: np.linalg.norm(
                               np.array(global_goal[:2]) - np.array(x[1]["center"][:2])))[0]
        print(f"⚠ 목표점 → 가장 가까운 방: {goal_room_id}")

    # 방 시퀀스 (BFS)
    path_rooms = find_path_between_rooms(start_room_id, goal_room_id, graph_data)
    print(f"\n방 경로 ({len(path_rooms)}개): {' → '.join(path_rooms)}")

    # 실제 A* 경로 계산
    segments = None
    if npy_dir and PATHFINDER_AVAILABLE:
        print(f"\n[방별 A* 경로 계산 중...]")
        segments = compute_actual_paths(graph_data, path_rooms,
                                        global_start, global_goal,
                                        npy_dir, resolution, sample, margin)
        if segments:
            total_wp = sum(len(s["waypoints"]) for s in segments)
            n_astar = sum(1 for s in segments if s["method"] == "A*")
            print(f"\n총 {len(segments)}개 방, A* 성공 {n_astar}개, "
                  f"전체 waypoints {total_wp}개")
    else:
        print(f"\n[A* 건너뜀 — 직선만 표시]")

    # ── 그림 생성 ─────────────────────────────────────────────────
    n_floors = len(floors)
    fig, axes = plt.subplots(1, n_floors, figsize=(7.5 * n_floors, 8))
    if n_floors == 1:
        axes = [axes]

    for idx, floor_num in enumerate(sorted(floors.keys())):
        ax = axes[idx]
        floor_rooms = {rid: ri for rid, ri in floors[floor_num]}

        ax.set_facecolor("#fafafa")
        ax.set_aspect("equal")

        # 1) 방 그리기
        for room_id, room_info in floor_rooms.items():
            bb_min = np.array(room_info["bbox_min"][:2])
            bb_max = np.array(room_info["bbox_max"][:2])

            is_start = (room_id == start_room_id)
            is_goal = (room_id == goal_room_id)
            is_path = (room_id in path_rooms)

            if is_start:
                fill, ec, lw = COLOR_START_FILL, COLOR_START_EDGE, 2.5
            elif is_goal:
                fill, ec, lw = COLOR_GOAL_FILL, COLOR_GOAL_EDGE, 2.5
            elif is_path:
                fill, ec, lw = COLOR_PATH_FILL, COLOR_PATH_EDGE, 2.0
            else:
                fill, ec, lw = COLOR_OTHER_FILL, COLOR_OTHER_EDGE, 0.8

            rect = Rectangle(bb_min, bb_max[0] - bb_min[0], bb_max[1] - bb_min[1],
                             linewidth=lw, edgecolor=ec, facecolor=fill, alpha=0.55)
            ax.add_patch(rect)

            # 방 ID 라벨
            cx, cy = (bb_min + bb_max) / 2
            ax.text(cx, cy, room_id, fontsize=9, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.9, edgecolor=ec, linewidth=0.8))

        # 2) 문/통로 마커
        for edge in edges:
            a, b = edge["a"], edge["b"]
            if a not in floor_rooms or b not in floor_rooms:
                continue
            door = np.array(edge["door_center"][:2])
            etype = edge.get("type", "door")

            is_path_edge = (a in path_rooms and b in path_rooms and
                            abs(path_rooms.index(a) - path_rooms.index(b)) == 1)

            mc = COLOR_DOOR_PATH if is_path_edge else COLOR_DOOR_OTHER
            sz = 10 if is_path_edge else 6
            zo = 5 if is_path_edge else 4

            if etype == "stairs":
                ax.plot(door[0], door[1], 's', color=mc, markersize=sz + 1,
                        markeredgecolor="black", markeredgewidth=0.8,
                        alpha=0.95, zorder=zo)
            else:
                ax.plot(door[0], door[1], 'D', color=mc, markersize=sz,
                        markeredgecolor="black", markeredgewidth=0.5,
                        alpha=0.9, zorder=zo)

        # 3) 층 간 stairs 표시 (이 층에서 다른 층으로)
        for edge in edges:
            if edge.get("type") != "stairs":
                continue
            a, b = edge["a"], edge["b"]
            a_floor = rooms[a]["floor"] if a in rooms else None
            b_floor = rooms[b]["floor"] if b in rooms else None

            if floor_num not in (a_floor, b_floor) or a_floor == b_floor:
                continue
            this_room = a if a in floor_rooms else (b if b in floor_rooms else None)
            if not this_room:
                continue

            door = np.array(edge["door_center"][:2])
            other = b_floor if a_floor == floor_num else a_floor
            direction = "↑" if other > floor_num else "↓"

            is_path_edge = (a in path_rooms and b in path_rooms and
                            abs(path_rooms.index(a) - path_rooms.index(b)) == 1)
            color = COLOR_DOOR_PATH if is_path_edge else COLOR_DOOR_OTHER
            fw = "bold" if is_path_edge else "normal"

            ax.annotate(f"{direction}F{other}", xy=(door[0], door[1]),
                        xytext=(door[0] + 0.4, door[1] + 0.4),
                        fontsize=8, fontweight=fw, color=color, zorder=6)

        # 4) 실제 A* 경로 그리기 (이 층의 segments만)
        if segments:
            floor_segs = [s for s in segments if s["floor"] == floor_num]
            for seg in floor_segs:
                wp = np.array(seg["waypoints"])

                # 메인 경로 라인
                ax.plot(wp[:, 0], wp[:, 1], '-', color=COLOR_PATH_LINE,
                        linewidth=2.8, alpha=0.95, zorder=7,
                        solid_capstyle='round', solid_joinstyle='round')

                # waypoint 점 (중간 점들만)
                if len(wp) > 2:
                    ax.plot(wp[1:-1, 0], wp[1:-1, 1], 'o',
                            color=COLOR_PATH_LINE, markersize=3.5, alpha=0.7,
                            zorder=8, markeredgecolor='white', markeredgewidth=0.5)

                # 진행 방향 화살표 (경로 중간 1~2개)
                if len(wp) >= 2:
                    n_arrows = max(1, min(2, len(wp) // 3))
                    indices = np.linspace(0, len(wp) - 2, n_arrows, dtype=int)
                    for ai in indices:
                        p1, p2 = wp[ai, :2], wp[ai + 1, :2]
                        dx, dy = p2 - p1
                        if dx ** 2 + dy ** 2 < 0.04:
                            continue
                        mid = (p1 + p2) / 2
                        ax.annotate('',
                                    xy=mid + np.array([dx, dy]) * 0.15,
                                    xytext=mid - np.array([dx, dy]) * 0.05,
                                    arrowprops=dict(arrowstyle='->',
                                                    color=COLOR_PATH_LINE,
                                                    lw=2.5, alpha=0.9),
                                    zorder=8)
        else:
            # 폴백: 방 중심 → 문 → 방 중심 (직선)
            for i in range(len(path_rooms) - 1):
                a, b = path_rooms[i], path_rooms[i + 1]
                if a not in floor_rooms or b not in floor_rooms:
                    continue
                door, _ = find_door_between(a, b, edges)
                if door is None:
                    continue
                ca = (np.array(rooms[a]["bbox_min"][:2]) +
                      np.array(rooms[a]["bbox_max"][:2])) / 2
                cb = (np.array(rooms[b]["bbox_min"][:2]) +
                      np.array(rooms[b]["bbox_max"][:2])) / 2
                ax.plot([ca[0], door[0], cb[0]], [ca[1], door[1], cb[1]],
                        color=COLOR_PATH_LINE, linewidth=2.5, alpha=0.85, zorder=7)

        # 5) 시작/목표점 별표
        if start_room_id in floor_rooms:
            sp = np.array(global_start[:2])
            ax.plot(sp[0], sp[1], '*', color=COLOR_START_EDGE, markersize=24,
                    zorder=10, markeredgecolor="black", markeredgewidth=1.2)
            ax.annotate(" 시작", xy=(sp[0], sp[1]), fontsize=11,
                        fontweight="bold", color=COLOR_START_EDGE, zorder=11)
        if goal_room_id in floor_rooms:
            gp = np.array(global_goal[:2])
            ax.plot(gp[0], gp[1], '*', color=COLOR_GOAL_EDGE, markersize=24,
                    zorder=10, markeredgecolor="black", markeredgewidth=1.2)
            ax.annotate(" 목표", xy=(gp[0], gp[1]), fontsize=11,
                        fontweight="bold", color=COLOR_GOAL_EDGE, zorder=11)

        # 6) 축
        all_min = np.min([np.array(r["bbox_min"][:2]) for r in floor_rooms.values()], axis=0)
        all_max = np.max([np.array(r["bbox_max"][:2]) for r in floor_rooms.values()], axis=0)
        ax.set_xlim(all_min[0] - 1, all_max[0] + 1)
        ax.set_ylim(all_min[1] - 1, all_max[1] + 1)
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_title(f"Floor {floor_num}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.15)

    # 전체 제목
    method_str = "실제 A* 경로" if (segments and any(s["method"] == "A*" for s in segments)) else "방 시퀀스"
    fig.suptitle(
        f"건축 평면도 + {method_str}\n"
        f"시작 {tuple(global_start)} → 목표 {tuple(global_goal)}   |   "
        f"방 경로: {' → '.join(path_rooms)}",
        fontsize=13, fontweight="bold", y=0.99
    )

    # 범례
    legend_elements = [
        mpatches.Patch(facecolor=COLOR_START_FILL, edgecolor=COLOR_START_EDGE,
                       label="시작 방", linewidth=2),
        mpatches.Patch(facecolor=COLOR_GOAL_FILL, edgecolor=COLOR_GOAL_EDGE,
                       label="목표 방", linewidth=2),
        mpatches.Patch(facecolor=COLOR_PATH_FILL, edgecolor=COLOR_PATH_EDGE,
                       label="경유 방", linewidth=2),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_DOOR_PATH,
                   markersize=9, label="문 (경로상)", markeredgecolor='black'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=COLOR_DOOR_PATH,
                   markersize=10, label="계단 (경로상)", markeredgecolor='black'),
        plt.Line2D([0], [0], color=COLOR_PATH_LINE, linewidth=2.8,
                   label="실제 경로"),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_PATH_LINE,
                   markersize=6, label="Waypoint"),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=COLOR_START_EDGE,
                   markersize=14, label="시작/목표", markeredgecolor='black'),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=8, fontsize=9,
               bbox_to_anchor=(0.5, -0.005), frameon=True)

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n✓ 저장: {Path(output_path).resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="건축 평면도 + 실제 A* 경로 시각화")
    parser.add_argument("--rooms-json", type=Path, default=Path("rooms_graph.json"))
    parser.add_argument("--npy-dir", type=Path, default=Path("npy"),
                        help="방별 npy 디렉토리 (A* 실행용)")
    parser.add_argument("--global-start", type=float, nargs=3,
                        default=[-1.5, -4.0, 0.5])
    parser.add_argument("--global-goal", type=float, nargs=3,
                        default=[14.0, 6.5, 4.0])
    parser.add_argument("--output", type=Path,
                        default=Path("floorplan_with_path.png"))
    parser.add_argument("--resolution", type=float, default=0.15)
    parser.add_argument("--sample", type=int, default=10)
    parser.add_argument("--margin", type=int, default=0)
    parser.add_argument("--no-astar", action="store_true",
                        help="A* 안 돌리고 직선으로만 표시 (빠름)")
    args = parser.parse_args()

    print("=" * 70)
    print("  건축 평면도 + 실제 A* 경로 시각화")
    print("=" * 70)
    print(f"  JSON:    {args.rooms_json}")
    print(f"  NPY:     {args.npy_dir if not args.no_astar else '(건너뜀)'}")
    print(f"  시작:    {tuple(args.global_start)}")
    print(f"  목표:    {tuple(args.global_goal)}")
    print(f"  A*:      {'OFF (직선)' if args.no_astar else 'ON'}")
    print(f"  Output:  {args.output}")
    print("=" * 70)

    visualize(
        args.rooms_json,
        args.global_start,
        args.global_goal,
        args.output,
        npy_dir=None if args.no_astar else args.npy_dir,
        resolution=args.resolution,
        sample=args.sample,
        margin=args.margin,
    )
