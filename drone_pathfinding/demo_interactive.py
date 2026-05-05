"""
demo_interactive.py - 인터랙티브 데모

실행: python demo_interactive.py

조작:
  [모드 전환]
  'p' → 포인트 모드 (기본): 왼클릭=시작점, 우클릭=목표점
  'd' → 장애물 그리기 모드: 클릭 & 드래그로 장애물 추가
  'e' → 지우개 모드: 클릭 & 드래그로 장애물 제거

  [브러시 크기]
  '1'~'5' → 브러시 크기 변경 (1=작게, 5=크게)

  [알고리즘]
  'a' → A*
  'r' → RRT*
  's' → PID 추종 시뮬레이션 재생

  [기타]
  'c' → 경로/시작/목표 초기화
  'x' → 맵 전체 초기화 (장애물 포함)
  'q' → 종료
"""

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import time

from maps import GridMap2D
from astar import AStarPlanner
from rrt import RRTStarPlanner
from core import Controller


class InteractiveDemo:
    def __init__(self):
        self.map_ = self._create_map()
        self.start = None
        self.goal = None
        self.path = []
        self.algorithm = 'astar'
        self.astar = AStarPlanner()
        self.rrt = RRTStarPlanner(max_iter=5000, step_size=3.0)

        # 모드: 'point', 'draw', 'erase'
        self.mode = 'point'
        self.brush_size = 2
        self.is_dragging = False

        # matplotlib 설정
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._draw_map()
        self._update_title()
        self._print_help()
        plt.show()

    def _print_help(self):
        print("\n╔══════════════════════════════════════════╗")
        print("║    Drone Pathfinding Interactive Demo     ║")
        print("╠══════════════════════════════════════════╣")
        print("║  MODE                                    ║")
        print("║  'p' → Point (start/goal)                ║")
        print("║  'd' → Draw obstacles (click & drag)     ║")
        print("║  'e' → Erase obstacles (click & drag)    ║")
        print("║                                           ║")
        print("║  BRUSH: '1'~'5'                           ║")
        print("║  ALGO:  'a'=A*  'r'=RRT*  's'=simulate   ║")
        print("║  RESET: 'c'=path  'x'=all  'q'=quit      ║")
        print("╚══════════════════════════════════════════╝")

    def _create_map(self) -> GridMap2D:
        m = GridMap2D(60, 60, resolution=0.5)
        # 기본 장애물 약간만 (자유롭게 수정 가능)
        m.add_obstacle_rect(15, 10, 3, 20)
        m.add_obstacle_rect(30, 25, 3, 25)
        m.add_obstacle_circle(45, 15, 4)
        m.add_obstacle_circle(20, 45, 3)
        return m

    def _draw_map(self):
        self.ax.clear()
        self.ax.imshow(self.map_.grid, cmap='Greys', origin='lower', alpha=0.7)
        self.ax.set_xlim(-1, self.map_.width)
        self.ax.set_ylim(-1, self.map_.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.2)

    def _update_title(self):
        algo_name = "A*" if self.algorithm == 'astar' else "RRT*"
        mode_name = {'point': 'Point (start/goal)',
                     'draw': f'Draw obstacle (brush={self.brush_size})',
                     'erase': f'Erase obstacle (brush={self.brush_size})'}
        status = f"Mode: {mode_name[self.mode]}  |  Algo: {algo_name}"
        if self.path:
            status += f"  |  Path: {len(self.path)} waypoints"
        self.fig.suptitle(status, fontsize=12, fontweight='bold')

    # ── 장애물 브러시 ──

    def _paint(self, x, y):
        r = self.brush_size
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px, py = x + dx, y + dy
                    if 0 <= px < self.map_.width and 0 <= py < self.map_.height:
                        if self.mode == 'draw':
                            self.map_.grid[py, px] = 1
                        elif self.mode == 'erase':
                            self.map_.grid[py, px] = 0

    # ── 마우스 이벤트 ──

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))

        if self.mode == 'point':
            if not self.map_.is_free((x, y)):
                print(f"  ({x}, {y}) is obstacle!")
                return
            if event.button == 1:
                self.start = (x, y)
                print(f"  Start: ({x}, {y})")
            elif event.button == 3:
                self.goal = (x, y)
                print(f"  Goal: ({x}, {y})")
            self._redraw()
            if self.start and self.goal:
                self._find_path()

        elif self.mode in ('draw', 'erase'):
            self.is_dragging = True
            self._paint(x, y)
            self._redraw()

    def _on_release(self, event):
        if self.is_dragging:
            self.is_dragging = False
            # 장애물 변경 후 경로 재탐색
            if self.start and self.goal:
                self._find_path()

    def _on_motion(self, event):
        if not self.is_dragging or event.inaxes != self.ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))
        self._paint(x, y)
        self._redraw()

    # ── 키보드 이벤트 ──

    def _on_key(self, event):
        if event.key == 'p':
            self.mode = 'point'
            print("\n  >> Point mode (left=start, right=goal)")
        elif event.key == 'd':
            self.mode = 'draw'
            print(f"\n  >> Draw obstacle mode (brush={self.brush_size})")
        elif event.key == 'e':
            self.mode = 'erase'
            print(f"\n  >> Erase obstacle mode (brush={self.brush_size})")
        elif event.key in '12345':
            self.brush_size = int(event.key)
            print(f"  >> Brush size: {self.brush_size}")
        elif event.key == 'a':
            self.algorithm = 'astar'
            print("\n  >> A* selected")
            if self.start and self.goal:
                self._find_path()
        elif event.key == 'r':
            self.algorithm = 'rrt'
            print("\n  >> RRT* selected")
            if self.start and self.goal:
                self._find_path()
        elif event.key == 's':
            if self.path:
                self._simulate_pid()
            else:
                print("  >> No path. Set start & goal first.")
        elif event.key == 'c':
            self.start = None
            self.goal = None
            self.path = []
            self._redraw()
            print("\n  >> Path reset (obstacles kept)")
        elif event.key == 'x':
            self.map_.grid[:] = 0
            self.start = None
            self.goal = None
            self.path = []
            self._redraw()
            print("\n  >> Full reset (map cleared)")
        elif event.key == 'q':
            plt.close()
            return

        self._update_title()
        self.fig.canvas.draw()

    # ── 경로 탐색 ──

    def _find_path(self):
        if self.start and not self.map_.is_free(self.start):
            print("  >> Start is now inside obstacle, cleared.")
            self.start = None
        if self.goal and not self.map_.is_free(self.goal):
            print("  >> Goal is now inside obstacle, cleared.")
            self.goal = None
        if not self.start or not self.goal:
            self.path = []
            self._redraw()
            return

        print(f"  Searching ({self.algorithm})...", end=' ')
        t0 = time.time()

        if self.algorithm == 'astar':
            self.path = self.astar.plan(self.map_, self.start, self.goal)
        else:
            self.path = self.rrt.plan(self.map_, self.start, self.goal)

        elapsed = time.time() - t0
        if self.path:
            print(f"Found! {len(self.path)} waypoints, {elapsed:.3f}s")
        else:
            print(f"No path found. {elapsed:.3f}s")

        self._redraw()

    def _redraw(self):
        self._draw_map()

        if self.algorithm == 'astar' and hasattr(self.astar, 'explored_nodes') and self.astar.explored_nodes:
            ex = np.array(self.astar.explored_nodes)
            self.ax.scatter(ex[:, 0], ex[:, 1], c='lightblue', s=3, alpha=0.3)
        if self.algorithm == 'rrt' and hasattr(self.rrt, 'tree_edges') and self.rrt.tree_edges:
            for (a, b) in self.rrt.tree_edges:
                self.ax.plot([a[0], b[0]], [a[1], b[1]], 'c-', alpha=0.1, linewidth=0.5)

        if self.path:
            px = [p[0] for p in self.path]
            py = [p[1] for p in self.path]
            self.ax.plot(px, py, 'r-', linewidth=2.5)
            self.ax.scatter(px, py, c='red', s=15, zorder=5)

        if self.start:
            self.ax.scatter(*self.start, c='lime', s=300, marker='*', zorder=10, edgecolors='black')
        if self.goal:
            self.ax.scatter(*self.goal, c='gold', s=300, marker='*', zorder=10, edgecolors='black')

        self._update_title()
        self.fig.canvas.draw()

    def _simulate_pid(self):
        print("  >> PID simulation starting...")
        controller = Controller(
            waypoints=self.path, kp=1.5, ki=0.01, kd=0.5, arrival_threshold=0.8
        )
        pos = np.array(self.start, dtype=float)
        dt, max_speed = 0.1, 5.0

        drone_dot, = self.ax.plot([], [], 'bo', markersize=10, zorder=15)
        trail_line, = self.ax.plot([], [], 'b-', linewidth=2, alpha=0.7)
        trail_x, trail_y = [], []

        for step in range(2000):
            if controller.is_done:
                break
            velocity = controller.compute(tuple(pos), dt)
            speed = np.linalg.norm(velocity)
            if speed > max_speed:
                velocity = velocity / speed * max_speed
            pos += velocity * dt
            trail_x.append(pos[0])
            trail_y.append(pos[1])
            if step % 5 == 0:
                drone_dot.set_data([pos[0]], [pos[1]])
                trail_line.set_data(trail_x, trail_y)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.01)

        drone_dot.set_data([pos[0]], [pos[1]])
        trail_line.set_data(trail_x, trail_y)
        self.fig.canvas.draw()
        print(f"  >> Done! {len(trail_x)} steps")


if __name__ == '__main__':
    InteractiveDemo()