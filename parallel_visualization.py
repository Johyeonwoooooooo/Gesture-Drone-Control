#!/usr/bin/env python3
"""
방별 npy 파일 시각화 (2D top-down).
- 3개 패널: 000_xxx, 001_xxx, 002_xxx 그룹별
- 각 방 점군 + 방 번호 라벨 + 중심점 X 마크
- 같은 그룹 내 인접 방 → 초록색 엣지
- 다른 그룹 간 인접 방 → 주황색 엣지 (계단/통로)
- 전역 시작/목표 강조 표시
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# 방 로드
# ──────────────────────────────────────────────────────────────────────────────
def load_rooms(npy_dir):
    folders = sorted(glob.glob(str(npy_dir / "*")))
    folders = [f for f in folders if os.path.isdir(f)]

    rooms = []
    for folder in folders:
        coord_path = os.path.join(folder, "coord.npy")
        if not os.path.exists(coord_path):
            continue

        points = np.load(coord_path).astype(float)
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        points = points[:, :3]

        name = os.path.basename(folder).replace("00809_Qpor2mEya8F_", "")
        # 000_002 → group="000", room_id="002"
        parts = name.split("_")
        group_id = parts[0]
        room_id = parts[1]

        rooms.append({
            "name": name,
            "group": group_id,
            "room_id": room_id,
            "points": points,
            "center": points.mean(axis=0),
            "xyz_min": points.min(axis=0),
            "xyz_max": points.max(axis=0),
        })
    return rooms


# ──────────────────────────────────────────────────────────────────────────────
# 인접 관계 계산 (bbox 거리 기준)
# ──────────────────────────────────────────────────────────────────────────────
def compute_adjacency(rooms, max_dist=2.0):
    """방들 사이 인접 엣지 추출. (i, j, is_inter_group) 튜플 리스트 반환"""
    edges = []
    n = len(rooms)
    for i in range(n):
        for j in range(i + 1, n):
            r1, r2 = rooms[i], rooms[j]

            # X-Y 평면에서 bounding box 사이 거리 계산
            dx = max(0, max(r1["xyz_min"][0] - r2["xyz_max"][0],
                            r2["xyz_min"][0] - r1["xyz_max"][0]))
            dy = max(0, max(r1["xyz_min"][1] - r2["xyz_max"][1],
                            r2["xyz_min"][1] - r1["xyz_max"][1]))
            dist_xy = np.sqrt(dx * dx + dy * dy)

            # Z(높이)도 너무 차이 나면 다른 층이라 연결 안 함
            z_overlap = (min(r1["xyz_max"][2], r2["xyz_max"][2]) -
                         max(r1["xyz_min"][2], r2["xyz_min"][2]))

            if dist_xy < max_dist and z_overlap > -1.0:
                is_inter = (r1["group"] != r2["group"])
                edges.append((i, j, is_inter))
    return edges


def find_room_containing(rooms, point):
    """주어진 좌표가 어느 방 bbox 안에 있는지"""
    p = np.array(point)
    for i, r in enumerate(rooms):
        if np.all(r["xyz_min"] <= p) and np.all(p <= r["xyz_max"]):
            return i
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────────
def visualize(rooms, edges, global_start, global_goal, output):
    # 그룹별로 분리
    groups = sorted(set(r["group"] for r in rooms))
    # 그룹별 색상 (이미지와 비슷하게)
    group_colors = {
        "000": "#5b9bd5",  # 파랑
        "001": "#70ad47",  # 초록
        "002": "#c19a6b",  # 갈색
    }
    group_titles = {
        "000": "Group 000 (Floor 1-A)",
        "001": "Group 001 (Floor 1-B)",
        "002": "Group 002 (Floor 2)",
    }

    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 8))
    if n_groups == 1:
        axes = [axes]

    # 전체 X 범위 (패널들 사이에 엣지 그리기 위해)
    start_idx = find_room_containing(rooms, global_start)
    goal_idx = find_room_containing(rooms, global_goal)

    # 그룹별 패널 인덱스 매핑
    group_to_ax = {g: axes[i] for i, g in enumerate(groups)}

    for gi, group in enumerate(groups):
        ax = axes[gi]
        group_rooms = [r for r in rooms if r["group"] == group]
        color = group_colors.get(group, "#888888")

        # 1. 점군 (X-Y top-down)
        for r in group_rooms:
            ax.scatter(r["points"][:, 0], r["points"][:, 1],
                       s=0.3, c=color, alpha=0.4)

        # 2. 그룹 내부 엣지 (초록)
        for i, j, is_inter in edges:
            if is_inter:
                continue
            r1, r2 = rooms[i], rooms[j]
            if r1["group"] != group:
                continue
            ax.plot([r1["center"][0], r2["center"][0]],
                    [r1["center"][1], r2["center"][1]],
                    color="#2ca02c", linewidth=2.0, alpha=0.75, zorder=3)

        # 3. 방 중심점 X 마크 + 라벨
        for r in group_rooms:
            cx, cy = r["center"][0], r["center"][1]
            ax.scatter(cx, cy, marker="x", s=80, c="darkred",
                       linewidths=2, zorder=5)
            ax.annotate(r["room_id"],
                        xy=(cx, cy), xytext=(8, 8),
                        textcoords="offset points",
                        fontsize=11, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  fc="white", ec="black", alpha=0.9),
                        zorder=6)

        # 4. 시작/목표 강조
        if start_idx is not None and rooms[start_idx]["group"] == group:
            r = rooms[start_idx]
            ax.scatter(r["center"][0], r["center"][1],
                       s=400, marker="o", facecolor="lime",
                       edgecolor="darkgreen", linewidths=2.5, zorder=10)
            ax.annotate("START",
                        xy=(r["center"][0], r["center"][1]),
                        xytext=(15, -20), textcoords="offset points",
                        fontsize=12, fontweight="bold", color="darkgreen",
                        zorder=11)

        if goal_idx is not None and rooms[goal_idx]["group"] == group:
            r = rooms[goal_idx]
            ax.scatter(r["center"][0], r["center"][1],
                       s=500, marker="*", facecolor="red",
                       edgecolor="darkred", linewidths=2.5, zorder=10)
            ax.annotate("GOAL",
                        xy=(r["center"][0], r["center"][1]),
                        xytext=(15, -20), textcoords="offset points",
                        fontsize=12, fontweight="bold", color="darkred",
                        zorder=11)

        ax.set_title(group_titles.get(group, f"Group {group}"),
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("X (m)")
        if gi == 0:
            ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    # 5. 그룹 간 엣지 (주황) — 패널 사이를 잇는 선
    # figure 좌표계로 변환해서 그림
    inter_edges = [(i, j) for i, j, is_inter in edges if is_inter]
    for i, j in inter_edges:
        r1, r2 = rooms[i], rooms[j]
        ax1 = group_to_ax[r1["group"]]
        ax2 = group_to_ax[r2["group"]]

        # 두 점의 figure 좌표
        xy1 = ax1.transData.transform((r1["center"][0], r1["center"][1]))
        xy2 = ax2.transData.transform((r2["center"][0], r2["center"][1]))
        xy1_fig = fig.transFigure.inverted().transform(xy1)
        xy2_fig = fig.transFigure.inverted().transform(xy2)

        line = plt.Line2D([xy1_fig[0], xy2_fig[0]],
                          [xy1_fig[1], xy2_fig[1]],
                          transform=fig.transFigure,
                          color="#ff9933", linewidth=2.5, alpha=0.85,
                          zorder=1)
        fig.lines.append(line)

    fig.suptitle("방 인접 그래프 시각화 (방 점군 + 노드 + 엣지)",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--workers", type=int, default=4,
                        help="(호환성 용도, 시각화엔 사용 안 함)")
    parser.add_argument("--max-adj-dist", type=float, default=2.0,
                        help="인접 판정 거리(m)")
    parser.add_argument("--output", type=Path,
                        default=Path("room_visualization.png"))
    args = parser.parse_args()

    print("=" * 60)
    print("  방 시각화 (2D top-down + 인접 그래프)")
    print("=" * 60)
    print(f"  npy 폴더:     {args.npy_dir}")
    print(f"  전역 시작:    {tuple(args.global_start)}")
    print(f"  전역 목표:    {tuple(args.global_goal)}")
    print(f"  인접 임계:    {args.max_adj_dist}m")
    print("=" * 60 + "\n")

    rooms = load_rooms(args.npy_dir)
    if not rooms:
        print(f"[에러] {args.npy_dir} 에서 방 못 찾음"); return
    print(f"방 {len(rooms)}개 로드 완료")

    edges = compute_adjacency(rooms, max_dist=args.max_adj_dist)
    intra = sum(1 for *_, inter in edges if not inter)
    inter = sum(1 for *_, inter in edges if inter)
    print(f"인접 엣지 추출: 그룹 내부 {intra}개, 그룹 간 {inter}개")

    start_idx = find_room_containing(rooms, args.global_start)
    goal_idx = find_room_containing(rooms, args.global_goal)
    print(f"시작 방: {rooms[start_idx]['name'] if start_idx is not None else '(못 찾음)'}")
    print(f"목표 방: {rooms[goal_idx]['name'] if goal_idx is not None else '(못 찾음)'}\n")

    visualize(rooms, edges, args.global_start, args.global_goal, args.output)
    print(f"\n시각화 저장 완료: {args.output.resolve()}")


if __name__ == "__main__":
    main()