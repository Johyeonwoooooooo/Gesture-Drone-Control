"""
maps.py - 맵 구현체

지금 사용: GridMap2D
나중에 시뮬레이터 연결 시: VoxelMap3D 로 교체
"""

import numpy as np
from typing import List, Tuple
from core import Map, Point


class GridMap2D(Map):
    """
    2D 격자 맵
    0 = 빈 공간, 1 = 장애물
    """

    def __init__(self, width: int, height: int, resolution: float = 1.0):
        """
        width, height: 격자 크기
        resolution: 격자 1칸의 실제 크기 (미터)
                    시뮬레이터와 스케일 맞출 때 사용
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.zeros((height, width), dtype=np.int8)

    @property
    def dimensions(self) -> int:
        return 2

    def is_free(self, point: Point) -> bool:
        x, y = int(round(point[0])), int(round(point[1]))
        if not self.is_in_bounds((x, y)):
            return False
        return self.grid[y, x] == 0

    def is_in_bounds(self, point: Point) -> bool:
        x, y = int(round(point[0])), int(round(point[1]))
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(self, point: Point) -> List[Point]:
        x, y = int(point[0]), int(point[1])
        neighbors = []
        # 8방향 이동
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.is_in_bounds((nx, ny)) and self.is_free((nx, ny)):
                    # 대각선 이동 시 벽 끼임 방지
                    if dx != 0 and dy != 0:
                        if not self.is_free((x + dx, y)) or not self.is_free((x, y + dy)):
                            continue
                    neighbors.append((nx, ny))
        return neighbors

    def get_bounds(self) -> Tuple[Point, Point]:
        return (0, 0), (self.width - 1, self.height - 1)

    def add_obstacle_rect(self, x: int, y: int, w: int, h: int):
        """사각형 장애물 추가"""
        self.grid[y:y+h, x:x+w] = 1

    def add_obstacle_circle(self, cx: int, cy: int, radius: int):
        """원형 장애물 추가"""
        for y in range(max(0, cy - radius), min(self.height, cy + radius + 1)):
            for x in range(max(0, cx - radius), min(self.width, cx + radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    self.grid[y, x] = 1

    def to_world(self, grid_point: Point) -> Point:
        """격자 좌표 → 실제 좌표 (시뮬레이터 연결 시 사용)"""
        return tuple(c * self.resolution for c in grid_point)

    def from_world(self, world_point: Point) -> Point:
        """실제 좌표 → 격자 좌표"""
        return tuple(int(round(c / self.resolution)) for c in world_point)


class VoxelMap3D(Map):
    """
    3D 복셀 맵 - 시뮬레이터 연결 시 사용
    PLY → voxel 변환 후 이 클래스에 넣으면 됨
    
    사용 예시 (나중에):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud("scan.ply")
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.1)
        
        map3d = VoxelMap3D(width=100, height=100, depth=50, resolution=0.1)
        for voxel in voxel_grid.get_voxels():
            idx = voxel.grid_index
            map3d.grid[idx[2], idx[1], idx[0]] = 1
        
        # 그대로 같은 planner 사용 가능!
        path = astar.plan(map3d, start=(0,0,5), goal=(90,90,5))
    """

    def __init__(self, width: int, height: int, depth: int, resolution: float = 0.1):
        self.width = width
        self.height = height
        self.depth = depth
        self.resolution = resolution
        self.grid = np.zeros((depth, height, width), dtype=np.int8)

    @property
    def dimensions(self) -> int:
        return 3

    def is_free(self, point: Point) -> bool:
        x, y, z = int(round(point[0])), int(round(point[1])), int(round(point[2]))
        if not self.is_in_bounds((x, y, z)):
            return False
        return self.grid[z, y, x] == 0

    def is_in_bounds(self, point: Point) -> bool:
        x, y, z = int(round(point[0])), int(round(point[1])), int(round(point[2]))
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth

    def get_neighbors(self, point: Point) -> List[Point]:
        x, y, z = int(point[0]), int(point[1]), int(point[2])
        neighbors = []
        # 26방향 (8방향의 3D 확장)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    np_ = (x + dx, y + dy, z + dz)
                    if self.is_in_bounds(np_) and self.is_free(np_):
                        neighbors.append(np_)
        return neighbors

    def get_bounds(self) -> Tuple[Point, Point]:
        return (0, 0, 0), (self.width - 1, self.height - 1, self.depth - 1)

    def to_world(self, grid_point: Point) -> Point:
        return tuple(c * self.resolution for c in grid_point)

    def from_world(self, world_point: Point) -> Point:
        return tuple(int(round(c / self.resolution)) for c in world_point)