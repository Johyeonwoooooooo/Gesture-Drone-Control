"""
rrt.py - RRT* 경로 탐색

샘플링 기반이라 연속 공간에서 동작, 드론처럼 자유도가 높은 환경에 적합
차원에 무관: Map의 bounds와 is_free만 있으면 됨
"""

import math
import random
from typing import List, Optional, Tuple
from core import Map, Point, Planner


class RRTStarPlanner(Planner):
    """
    RRT* 알고리즘
    
    Parameters:
        max_iter: 최대 반복 횟수
        step_size: 한 번에 확장하는 거리
        goal_sample_rate: 목표점을 직접 샘플링할 확률 (0~1)
        search_radius: rewiring 탐색 반경 (None이면 자동 계산)
    """

    def __init__(self, max_iter: int = 3000, step_size: float = 2.0,
                 goal_sample_rate: float = 0.1, search_radius: Optional[float] = None):
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius

    def plan(self, map_: Map, start: Point, goal: Point) -> List[Point]:
        if not map_.is_free(start) or not map_.is_free(goal):
            print("[RRT*] 시작점 또는 목표점이 장애물 위에 있음")
            return []

        dims = map_.dimensions
        bounds_min, bounds_max = map_.get_bounds()

        # 트리: 각 노드는 (point, parent_idx, cost)
        nodes: List[Tuple[Point, Optional[int], float]] = [(start, None, 0.0)]

        # 시각화용
        self.tree_edges: List[Tuple[Point, Point]] = []

        goal_threshold = self.step_size * 1.5

        for i in range(self.max_iter):
            # 랜덤 샘플링 (가끔 목표점 직접 샘플)
            if random.random() < self.goal_sample_rate:
                sample = goal
            else:
                sample = tuple(
                    random.uniform(bounds_min[d], bounds_max[d])
                    for d in range(dims)
                )

            # 가장 가까운 노드 찾기
            nearest_idx = self._nearest(nodes, sample)
            nearest_point = nodes[nearest_idx][0]

            # step_size만큼 확장
            new_point = self._steer(nearest_point, sample)

            # 충돌 검사 (직선 경로)
            if not self._collision_free(map_, nearest_point, new_point):
                continue

            # RRT* rewiring: 반경 내 최적 부모 찾기
            radius = self.search_radius or self._compute_radius(len(nodes), dims)
            near_indices = self._near(nodes, new_point, radius)

            # 최소 비용 부모 선택
            best_parent = nearest_idx
            best_cost = nodes[nearest_idx][2] + self._dist(nearest_point, new_point)

            for idx in near_indices:
                node_point = nodes[idx][0]
                new_cost = nodes[idx][2] + self._dist(node_point, new_point)
                if new_cost < best_cost and self._collision_free(map_, node_point, new_point):
                    best_parent = idx
                    best_cost = new_cost

            # 노드 추가
            new_idx = len(nodes)
            nodes.append((new_point, best_parent, best_cost))
            self.tree_edges.append((nodes[best_parent][0], new_point))

            # 주변 노드 rewiring
            for idx in near_indices:
                node_point = nodes[idx][0]
                rewire_cost = best_cost + self._dist(new_point, node_point)
                if rewire_cost < nodes[idx][2] and self._collision_free(map_, new_point, node_point):
                    nodes[idx] = (node_point, new_idx, rewire_cost)

            # 목표 도달 확인
            if self._dist(new_point, goal) < goal_threshold:
                if self._collision_free(map_, new_point, goal):
                    nodes.append((goal, new_idx, best_cost + self._dist(new_point, goal)))
                    path = self._extract_path(nodes, len(nodes) - 1)
                    return self._smooth_path(map_, path)

        print(f"[RRT*] {self.max_iter}회 반복 후 경로를 찾지 못함")
        return []

    def _dist(self, a: Point, b: Point) -> float:
        return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))

    def _nearest(self, nodes, point):
        return min(range(len(nodes)), key=lambda i: self._dist(nodes[i][0], point))

    def _near(self, nodes, point, radius):
        return [i for i in range(len(nodes)) if self._dist(nodes[i][0], point) <= radius]

    def _steer(self, from_point: Point, to_point: Point) -> Point:
        d = self._dist(from_point, to_point)
        if d <= self.step_size:
            return to_point
        ratio = self.step_size / d
        return tuple(f + (t - f) * ratio for f, t in zip(from_point, to_point))

    def _collision_free(self, map_: Map, a: Point, b: Point, steps: int = 10) -> bool:
        """a→b 직선 경로에 장애물이 없는지 검사"""
        for i in range(steps + 1):
            t = i / steps
            point = tuple(ai + (bi - ai) * t for ai, bi in zip(a, b))
            if not map_.is_free(point):
                return False
        return True

    def _compute_radius(self, n: int, dims: int) -> float:
        """RRT* 이론적 최적 반경"""
        if n == 0:
            return self.step_size * 3
        return min(self.step_size * 3, 30.0 * (math.log(n + 1) / (n + 1)) ** (1.0 / dims))

    def _extract_path(self, nodes, goal_idx: int) -> List[Point]:
        path = []
        idx = goal_idx
        while idx is not None:
            path.append(nodes[idx][0])
            idx = nodes[idx][1]
        path.reverse()
        return path

    def _smooth_path(self, map_: Map, path: List[Point]) -> List[Point]:
        """불필요한 중간 waypoint 제거 (직선 연결 가능하면 생략)"""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            # 가능한 먼 점까지 직선 연결 시도
            j = len(path) - 1
            while j > i + 1:
                if self._collision_free(map_, path[i], path[j], steps=20):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j

        return smoothed