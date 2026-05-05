"""
simulator_bridge.py - 시뮬레이터 연결용 브릿지 (템플릿)

나중에 Unity 시뮬레이터가 완성되면 이 파일을 채워서 연결
현재는 인터페이스만 정의해둔 상태

연결 흐름:
  1. Unity에서 PLY 환경 로드 → VoxelMap3D 생성
  2. Unity에서 드론 위치 수신 → start 좌표
  3. Planner로 경로 생성 (A* or RRT*)
  4. Controller로 속도 명령 생성
  5. 속도 → rc 명령 변환 → Unity로 전송
"""

import socket
import json
import struct
from typing import Optional

from core import Controller, Point
from maps import VoxelMap3D
from astar import AStarPlanner
from rrt import RRTStarPlanner


# ============================================================
# 아래는 python_rc_test.py의 소켓 통신과 연결하는 예시
# 현우 브랜치의 TelloSimulator.cs와 맞춰서 수정 필요
# ============================================================

class SimulatorBridge:
    """Unity 시뮬레이터와 통신하는 브릿지"""

    def __init__(self, host: str = '127.0.0.1', port: int = 8889):
        self.host = host
        self.port = port
        self.sock: Optional[socket.socket] = None

    def connect(self):
        """시뮬레이터 연결"""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[Bridge] {self.host}:{self.port} 연결")

    def send_rc_command(self, lr: int, fb: int, ud: int, yaw: int):
        """
        rc 명령 전송 (python_rc_test.py 형식)
        lr: left/right (-100~100)
        fb: forward/backward (-100~100)
        ud: up/down (-100~100)
        yaw: rotation (-100~100)
        """
        cmd = f"rc {lr} {fb} {ud} {yaw}"
        if self.sock:
            self.sock.sendto(cmd.encode(), (self.host, self.port))

    def velocity_to_rc(self, velocity, max_speed: float = 5.0) -> tuple:
        """
        PID 컨트롤러 출력(velocity) → rc 명령 변환
        
        velocity: (vx, vy) for 2D or (vx, vy, vz) for 3D
        반환: (lr, fb, ud, yaw)
        """
        scale = 100.0 / max_speed

        if len(velocity) == 2:
            vx, vy = velocity
            lr = int(max(-100, min(100, vx * scale)))
            fb = int(max(-100, min(100, vy * scale)))
            return lr, fb, 0, 0
        elif len(velocity) == 3:
            vx, vy, vz = velocity
            lr = int(max(-100, min(100, vx * scale)))
            fb = int(max(-100, min(100, vy * scale)))
            ud = int(max(-100, min(100, vz * scale)))
            return lr, fb, ud, 0

    def get_drone_state(self) -> Optional[Point]:
        """
        TODO: Unity로부터 드론 현재 위치 수신
        TelloSimulator.cs에서 위치 데이터를 보내도록 수정 필요
        """
        pass

    def close(self):
        if self.sock:
            self.sock.close()


def autonomous_flight_example():
    """
    자율 비행 예시 (시뮬레이터 연결 후 사용)
    
    흐름:
      1. 맵 로드
      2. 경로 탐색
      3. 경로 추종 루프
    """

    # --- 1. 맵 로드 ---
    # TODO: PLY → VoxelMap3D 변환
    # map_3d = VoxelMap3D(100, 100, 50, resolution=0.1)
    # load_ply_to_voxelmap(map_3d, "scan.ply")

    # --- 2. 경로 탐색 ---
    # planner = AStarPlanner()  # 또는 RRTStarPlanner()
    # start = bridge.get_drone_state()
    # goal = (50, 50, 10)  # 목표 좌표
    # path = planner.plan(map_3d, start, goal)

    # --- 3. 경로 추종 ---
    # bridge = SimulatorBridge()
    # bridge.connect()
    # controller = Controller(path, kp=1.5, ki=0.01, kd=0.5)
    #
    # while not controller.is_done:
    #     current_pos = bridge.get_drone_state()
    #     velocity = controller.compute(current_pos, dt=0.05)
    #     rc = bridge.velocity_to_rc(velocity)
    #     bridge.send_rc_command(*rc)
    #     time.sleep(0.05)
    #
    # bridge.close()
    pass