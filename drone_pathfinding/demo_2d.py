"""
demo_2d.py - 2D 경로 탐색 시각화 데모

실행: python demo_2d.py
결과: A*와 RRT* 비교 이미지 저장

이 데모의 핵심 포인트:
  - 알고리즘은 Map 인터페이스에만 의존
  - GridMap2D → VoxelMap3D로 바꿔도 planner 코드 수정 없음
  - Controller가 출력하는 velocity를 rc 명령으로 매핑하면 시뮬레이터 연결 완료
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time

from maps import GridMap2D
from astar import AStarPlanner
from rrt import RRTStarPlanner
from core import Controller


def create_test_map() -> GridMap2D:
    """테스트용 2D 맵 생성"""
    m = GridMap2D(60, 60, resolution=0.5)  # 60x60 격자, 1칸 = 0.5m

    # 장애물 배치 (실내 환경 시뮬레이션)
    m.add_obstacle_rect(10, 0, 3, 35)    # 왼쪽 벽
    m.add_obstacle_rect(10, 40, 3, 20)   # 왼쪽 벽 (문 있음)
    m.add_obstacle_rect(25, 15, 3, 45)   # 중간 벽
    m.add_obstacle_rect(40, 0, 3, 30)    # 오른쪽 벽
    m.add_obstacle_rect(40, 38, 3, 22)   # 오른쪽 벽 (문 있음)

    # 원형 장애물 (기둥/가구)
    m.add_obstacle_circle(18, 10, 3)
    m.add_obstacle_circle(35, 50, 4)
    m.add_obstacle_circle(50, 15, 3)

    return m


def visualize_result(ax, map_: GridMap2D, path, title,
                     start, goal, explored=None, tree_edges=None):
    """경로 탐색 결과 시각화"""
    # 맵 그리기
    ax.imshow(map_.grid, cmap='Greys', origin='lower', alpha=0.7)

    # A* 탐색 영역 표시
    if explored:
        ex = np.array(explored)
        ax.scatter(ex[:, 0], ex[:, 1], c='lightblue', s=2, alpha=0.3, label='탐색 영역')

    # RRT* 트리 표시
    if tree_edges:
        for (a, b) in tree_edges:
            ax.plot([a[0], b[0]], [a[1], b[1]], 'c-', alpha=0.1, linewidth=0.5)

    # 경로 표시
    if path:
        px = [p[0] for p in path]
        py = [p[1] for p in path]
        ax.plot(px, py, 'r-', linewidth=2.5, label=f'경로 ({len(path)} waypoints)')
        ax.scatter(px, py, c='red', s=15, zorder=5)

    # 시작/끝점
    ax.scatter(*start, c='lime', s=200, marker='*', zorder=10, edgecolors='black', label='시작')
    ax.scatter(*goal, c='gold', s=200, marker='*', zorder=10, edgecolors='black', label='목표')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-1, map_.width)
    ax.set_ylim(-1, map_.height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)


def simulate_drone_movement(path, start, dt=0.1, max_steps=2000):
    """
    PID 컨트롤러로 드론이 경로를 따라가는 시뮬레이션
    
    반환: 실제 이동 궤적 (부드러운 곡선)
    나중에 시뮬레이터 연결 시:
        velocity → rc_command 변환만 추가하면 됨
    """
    controller = Controller(
        waypoints=path,
        kp=1.5, ki=0.01, kd=0.5,
        arrival_threshold=0.8
    )

    pos = np.array(start, dtype=float)
    trajectory = [pos.copy()]

    for _ in range(max_steps):
        if controller.is_done:
            break

        velocity = controller.compute(tuple(pos), dt)

        # 속도 제한 (드론 최대 속도 시뮬레이션)
        max_speed = 5.0
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            velocity = velocity / speed * max_speed

        pos += velocity * dt
        trajectory.append(pos.copy())

    return trajectory


def main():
    map_ = create_test_map()
    start = (3, 3)
    goal = (55, 55)

    # --- A* ---
    print("A* 탐색 중...")
    astar = AStarPlanner()
    t0 = time.time()
    path_astar = astar.plan(map_, start, goal)
    t_astar = time.time() - t0
    print(f"  완료: {len(path_astar)} waypoints, {t_astar:.3f}초")

    # --- RRT* ---
    print("RRT* 탐색 중...")
    rrt = RRTStarPlanner(max_iter=5000, step_size=3.0, goal_sample_rate=0.1)
    t0 = time.time()
    path_rrt = rrt.plan(map_, start, goal)
    t_rrt = time.time() - t0
    print(f"  완료: {len(path_rrt)} waypoints, {t_rrt:.3f}초")

    # --- 시각화 ---
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # A* 결과
    visualize_result(
        axes[0], map_, path_astar,
        f"A* (waypoints: {len(path_astar)}, time: {t_astar:.3f}s)",
        start, goal,
        explored=astar.explored_nodes
    )

    # RRT* 결과
    visualize_result(
        axes[1], map_, path_rrt,
        f"RRT* (waypoints: {len(path_rrt)}, time: {t_rrt:.3f}s)",
        start, goal,
        tree_edges=rrt.tree_edges
    )

    # PID 경로 추종 시뮬레이션
    if path_astar:
        print("PID 컨트롤러로 경로 추종 시뮬레이션...")
        trajectory = simulate_drone_movement(path_astar, start)
        traj = np.array(trajectory)

        axes[2].imshow(map_.grid, cmap='Greys', origin='lower', alpha=0.7)
        # 계획된 경로
        px = [p[0] for p in path_astar]
        py = [p[1] for p in path_astar]
        axes[2].plot(px, py, 'r--', linewidth=1.5, alpha=0.5, label='계획 경로')
        # 실제 궤적
        axes[2].plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='PID 추종 궤적')
        axes[2].scatter(*start, c='lime', s=200, marker='*', zorder=10, edgecolors='black')
        axes[2].scatter(*goal, c='gold', s=200, marker='*', zorder=10, edgecolors='black')
        axes[2].set_title("PID Controller 경로 추종", fontsize=14, fontweight='bold')
        axes[2].legend(loc='upper right', fontsize=8)
        axes[2].set_xlim(-1, map_.width)
        axes[2].set_ylim(-1, map_.height)
        axes[2].set_aspect('equal')
        axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig('pathfinding_demo.png', dpi=150, bbox_inches='tight')
    print("\n결과 저장: pathfinding_demo.png")
    plt.close()


if __name__ == '__main__':
    main()