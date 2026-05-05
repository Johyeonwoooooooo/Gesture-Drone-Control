"""
astar.py - A* 경로 탐색

차원에 무관하게 동작: Map 인터페이스만 구현되어 있으면 2D/3D 모두 OK
"""

import heapq
import math
from typing import List, Dict, Optional
from core import Map, Point, Planner


def _heuristic(a: Point, b: Point) -> float:
    """유클리드 거리 (2D/3D 자동 대응)"""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _move_cost(a: Point, b: Point) -> float:
    """이동 비용: 직선 1.0, 대각선 sqrt(2), 3D 대각선 sqrt(3)"""
    return _heuristic(a, b)


class AStarPlanner(Planner):
    """
    A* 알고리즘
    
    사용법:
        planner = AStarPlanner()
        path = planner.plan(map_2d, start=(0,0), goal=(49,49))
        # 나중에 3D:
        path = planner.plan(map_3d, start=(0,0,5), goal=(90,90,5))
    """

    def plan(self, map_: Map, start: Point, goal: Point) -> List[Point]:
        if not map_.is_free(start):
            print(f"[A*] 시작점 {start} 이 장애물 위에 있음")
            return []
        if not map_.is_free(goal):
            print(f"[A*] 목표점 {goal} 이 장애물 위에 있음")
            return []

        # 정수 좌표로 정규화
        start = tuple(int(round(s)) for s in start)
        goal = tuple(int(round(g)) for g in goal)

        open_set = []  # (f_score, counter, point)
        counter = 0
        heapq.heappush(open_set, (_heuristic(start, goal), counter, start))

        came_from: Dict[Point, Point] = {}
        g_score: Dict[Point, float] = {start: 0}
        closed_set = set()

        # 탐색 과정 기록 (시각화용)
        self.explored_nodes: List[Point] = []

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)
            self.explored_nodes.append(current)

            # 도착
            if current == goal:
                return self._reconstruct(came_from, current)

            for neighbor in map_.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + _move_cost(current, neighbor)

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + _heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

        print("[A*] 경로를 찾을 수 없음")
        return []

    def _reconstruct(self, came_from: Dict[Point, Point], current: Point) -> List[Point]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path