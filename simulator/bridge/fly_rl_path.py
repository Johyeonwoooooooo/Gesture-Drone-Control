# -*- coding: utf-8 -*-
"""
fly_rl_path.py — 강화학습 경로를 시뮬레이터에서 실제로 비행시킨다
────────────────────────────────────────────────────────────────────
드론 현재 위치 -> (SAC 정책) -> 월드 궤적 -> 시야 단축 -> Unity 좌표 변환
-> takeoff -> 20Hz PID 추종 -> land 까지 한 번에.

Unity exe 가 없어도 지금 전부 검증할 수 있다. 프로토콜 스텁을 먼저 띄우면
된다(물리·포트가 TelloSimulator.cs 와 맞춰져 있다):

    python simulator/bridge/fake_unity_sim.py          # 터미널 1
    python simulator/bridge/fly_rl_path.py --room 016  # 터미널 2

exe 가 오면 스텁만 끄고 `--unity-host <노트북IP>` 로 바꾸면 그대로 돈다.

    python simulator/bridge/fly_rl_path.py --goal 6.0 1.0 4.2
    python simulator/bridge/fly_rl_path.py --room 016 --dry-run   # 계획만
    python simulator/bridge/fly_rl_path.py --room 002 --speed 3

출발점은 항상 **드론의 현재 위치**다(`--start` 로 강제 지정 가능). 시뮬레이터
상태 패킷의 Unity 좌표를 월드로 역변환해서 정책에 넣는다.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
for p in (_HERE, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import rl_planner                                    # noqa: E402
from coord_transform import load_building_transform  # noqa: E402
from unity_bridge import UnityTelloBridge            # noqa: E402
import follow_path as FP                             # noqa: E402

BUILDING = '00809_Qpor2mEya8F'
META = os.path.join(_ROOT, 'web', 'uploads', 'web_meta.json')


def room_center(room_id: str):
    """web_meta.json 의 방 중심을 목적지로. (편의용 - 좌표 외우기 싫을 때)"""
    if not os.path.exists(META):
        raise SystemExit(f'방 메타가 없습니다: {META}')
    meta = json.load(open(META, encoding='utf-8'))
    rm = meta['rooms'].get(room_id)
    if rm is None:
        raise SystemExit(f'그런 방이 없습니다: {room_id} '
                         f'(가능: {", ".join(sorted(meta["rooms"]))})')
    return np.array(rm['center'], dtype=float), rm['floor']


def main() -> int:
    ap = argparse.ArgumentParser(description='강화학습 경로로 시뮬레이터 비행')
    ap.add_argument('--goal', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                    help='도착 좌표 (월드 미터)')
    ap.add_argument('--room', help='도착 방 번호 (예: 016). --goal 대신 사용')
    ap.add_argument('--start', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                    help='출발 좌표 강제 지정 (기본: 드론 현재 위치)')
    ap.add_argument('--unity-host', default='127.0.0.1')
    ap.add_argument('--unity-port', type=int, default=9000)
    ap.add_argument('--local-port', type=int, default=9001)
    ap.add_argument('--state-port', type=int, default=9002)
    ap.add_argument('--speed', type=float, default=2.0,
                    help='Unity 단위 속도 u/s (집 스케일 5배라 2.0 = 0.4 m/s)')
    ap.add_argument('--rc-limit', type=int, default=30)
    ap.add_argument('--timeout', type=float, default=None)
    ap.add_argument('--no-smooth', action='store_true',
                    help='시야 단축 없이 0.3m 조밀 궤적 그대로 비행')
    ap.add_argument('--dry-run', action='store_true',
                    help='경로만 계산하고 비행하지 않음 (Unity 불필요)')
    ap.add_argument('--people', type=int, default=0, help='동적 장애물(사람) 수')
    ap.add_argument('--shield', action='store_true', help='런타임 안전막')
    a = ap.parse_args()

    if a.room:
        goal, floor = room_center(a.room)
        print(f'[목적지] 방 {a.room} (층 {floor}) 중심 = {np.round(goal, 2)}')
    elif a.goal:
        goal = np.asarray(a.goal, dtype=float)
    else:
        return ap.error('--goal 또는 --room 중 하나는 필요합니다') or 2

    xform = load_building_transform(BUILDING)
    bridge = None

    # ── 출발점: 드론 현재 위치 ─────────────────────────────────────
    if a.start:
        start = np.asarray(a.start, dtype=float)
        print(f'[출발] 지정 좌표 {np.round(start, 2)}')
    elif a.dry_run:
        start = rl_planner.home()
        print(f'[출발] dry-run - 홈 {np.round(start, 2)}')
    else:
        bridge = UnityTelloBridge(unity_host=a.unity_host, command_port=a.unity_port,
                                  local_command_port=a.local_port,
                                  local_state_port=a.state_port)
        bridge.connect()
        print(f'[연결] {a.unity_host}:{a.unity_port} 상태 대기…')
        st = bridge.wait_for_state(timeout=5.0)
        if st is None:
            print('[연결] 실패 - 시뮬레이터가 9000 을 듣고 있는지 확인하세요 '
                  '(Unity Play 또는 fake_unity_sim.py)')
            bridge.close()
            return 1
        start = xform.unity_to_mosaic(np.array([st.x, st.y, st.z], dtype=float))
        print(f'[출발] 드론 현재 위치 unity=({st.x:.2f}, {st.y:.2f}, {st.z:.2f}) '
              f'-> 월드 {np.round(start, 2)}')

    # ── 경로: 강화학습 정책 ────────────────────────────────────────
    rl_planner.load()
    wps, info = rl_planner.plan(start, goal, people=a.people, shield=a.shield,
                                smooth=not a.no_smooth)
    print(f"[경로] {'도달 성공' if info['success'] else '미도달(부분 궤적)'} | "
          f"{info['steps']}스텝 | 조밀 {len(info['raw'])}점 {info['raw_length_m']:.2f}m "
          f"-> waypoint {info['n_waypoints']}개 {info['length_m']:.2f}m | "
          f"범프 {info['bumps']} | {info['ms']:.0f}ms")
    if not np.allclose(info['start'], start, atol=1e-3):
        print(f"       출발 스냅: {np.round(start, 2)} -> {np.round(info['start'], 2)}")
    if not np.allclose(info['goal'], goal, atol=1e-3):
        print(f"       도착 스냅: {np.round(goal, 2)} -> {np.round(info['goal'], 2)}")
    if not info['success']:
        print('       ※ 정책이 목표에 도달하지 못했습니다. 그래도 간 데까지는 날립니다.')

    wps_unity = xform.mosaic_to_unity(np.asarray(wps, dtype=float))
    print(f'[변환] Unity 좌표 {len(wps_unity)}점, '
          f'{np.linalg.norm(np.diff(wps_unity, axis=0), axis=1).sum():.1f} u')

    if a.dry_run:
        print('[dry-run] 비행 생략')
        if bridge:
            bridge.close()
        return 0

    # ── 비행 ───────────────────────────────────────────────────────
    t0 = time.time()
    res = FP.fly_mission(
        bridge, [tuple(p) for p in wps_unity],
        setpos_start=False,                 # 드론이 서 있는 자리에서 그대로 출발
        max_speed=float(a.speed), rc_limit=int(a.rc_limit),
        timeout_sec=a.timeout)
    print(f'[비행] {"성공" if res.success else "중단"} ({res.reason}) | '
          f'최종오차 {res.final_error_u:.2f}u | 충돌 {res.collision_count} | '
          f'rc {res.rc_commands_sent}회 | {time.time() - t0:.0f}s')
    bridge.close()
    return 0 if res.success else 1


if __name__ == '__main__':
    sys.exit(main())
