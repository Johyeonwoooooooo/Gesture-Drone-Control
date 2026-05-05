"""
core.py - 차원에 의존하지 않는 인터페이스
2D든 3D든 이 인터페이스만 구현하면 알고리즘이 그대로 동작함

나중에 시뮬레이터 연결 시:
  - GridMap2D → VoxelMap3D 로 교체
  - Planner는 수정 없이 그대로 사용
  - Controller 출력을 rc 명령으로 매핑
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

# 좌표 타입: 2D면 (x,y), 3D면 (x,y,z)
Point = Tuple[float, ...]


class Map(ABC):
    """맵 인터페이스 - 2D grid든 3D voxel이든 이것만 구현하면 됨"""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """2 or 3"""
        pass

    @abstractmethod
    def is_free(self, point: Point) -> bool:
        """해당 좌표가 이동 가능한지"""
        pass

    @abstractmethod
    def is_in_bounds(self, point: Point) -> bool:
        """맵 범위 안인지"""
        pass

    @abstractmethod
    def get_neighbors(self, point: Point) -> List[Point]:
        """인접 이동 가능한 좌표들 (A* 등 grid 기반용)"""
        pass

    @abstractmethod
    def get_bounds(self) -> Tuple[Point, Point]:
        """맵의 (min_corner, max_corner) - RRT 샘플링 범위"""
        pass


class Planner(ABC):
    """경로 탐색 알고리즘 인터페이스"""

    @abstractmethod
    def plan(self, map_: Map, start: Point, goal: Point) -> List[Point]:
        """
        입력: 맵, 시작점, 목표점
        출력: waypoint 리스트 (시작 → 목표)
              경로 없으면 빈 리스트
        """
        pass


class Controller:
    """
    PID 기반 경로 추종 컨트롤러
    waypoint 리스트를 받아서 속도 명령을 출력
    
    나중에 시뮬레이터 연결 시:
      velocity = controller.compute(current_pos, dt)
      rc_command = velocity_to_rc(velocity)  # 매핑 함수만 추가
    """

    def __init__(self, waypoints: List[Point],
                 kp: float = 1.0, ki: float = 0.0, kd: float = 0.3,
                 arrival_threshold: float = 0.5):
        self.waypoints = list(waypoints)
        self.current_idx = 0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.arrival_threshold = arrival_threshold
        self._integral = np.zeros(len(waypoints[0]) if waypoints else 2)
        self._prev_error = np.zeros_like(self._integral)

    @property
    def target(self) -> Point:
        """현재 목표 waypoint"""
        if self.current_idx < len(self.waypoints):
            return self.waypoints[self.current_idx]
        return self.waypoints[-1]

    @property
    def is_done(self) -> bool:
        return self.current_idx >= len(self.waypoints)

    def compute(self, current_pos: Point, dt: float = 0.1) -> np.ndarray:
        """
        현재 위치 → 속도 명령 벡터 출력
        2D면 (vx, vy), 3D면 (vx, vy, vz)
        """
        if self.is_done:
            return np.zeros(len(current_pos))

        target = np.array(self.target, dtype=float)
        pos = np.array(current_pos, dtype=float)
        error = target - pos

        # 목표 waypoint 도달 시 다음으로
        if np.linalg.norm(error) < self.arrival_threshold:
            self.current_idx += 1
            self._integral *= 0  # reset
            self._prev_error *= 0
            return self.compute(current_pos, dt)

        # PID
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else np.zeros_like(error)
        self._prev_error = error.copy()

        velocity = self.kp * error + self.ki * self._integral + self.kd * derivative
        return velocity