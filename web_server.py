# -*- coding: utf-8 -*-
"""
web_server.py — 웹 관제 UI(HAUNTED OPS) + 강화학습 경로계획 백엔드
────────────────────────────────────────────────────────────────────
브라우저에서 평면도 위의 두 점(출발/도착)을 찍으면, 학습된 SAC 정책을
서버에서 실제로 굴려 3D 궤적을 만들어 돌려준다. 방그래프 BFS 나 RRT* 가
아니라 **강화학습 정책 단독 추론**이 경로를 만든다.

    conda activate tello
    python web_server.py                 # http://localhost:8000
    python web_server.py --port 9100
    python web_server.py --no-rl         # UI 만 (정책 없이 폴백 경로)

정적 파일은 최상위 web/ 를 그대로 서빙한다
(HAUNTED OPS.dc.html / 드론 관제.dc.html / support.js / uploads/).
playground/ 는 실험용이고, 실제로 돌리는 웹 앱은 web/ 에 있다.

출발점은 **드론의 현재 위치**로 고정이다. 웹에서는 도착지 한 점만 고른다.
`--unity-host` 를 주면 시뮬레이터의 실시간 위치를 쓰고(UDP 9002 상태 패킷을
Unity→월드 좌표로 역변환), 없으면 시뮬레이터 스폰 지점(HOME_WORLD)을 쓴다.

API
  GET  /api/status  →  {ready, engine, model, env}
  GET  /api/drone   →  {x,y,z, source:"sim"|"home", connected, flying}
  POST /plan        →  {start,goal,path,success,steps,bumps,dist,flown,ms,engine}
       body: {"goal":[x,y,z], "start":[x,y,z](생략 시 드론 현재 위치),
              "people":0, "shield":false}

주의: 모델 로드 + 환경 생성은 기동 시 1회(약 6초)만 하고 상주시킨다.
한 번의 경로 계획은 0.05~0.2초(최악 700스텝 ≈ 3.5초). 환경 객체는
재진입 불가라 Lock 으로 직렬화한다.
"""
import os
import sys
import time
import argparse
import threading

import numpy as _np
from flask import Flask, Response, request, jsonify, redirect, send_from_directory

import rl_planner               # 경로 계획(정책 로드/롤아웃/단축)은 전부 여기
import gesture_cam              # 웹캠 제스처 인식 + 미리보기 스트림

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(_ROOT, 'simulator', 'bridge')
_WEB = os.path.join(_ROOT, 'web')
_INDEX = 'HAUNTED OPS.dc.html'

sys.path.insert(0, _SIM)

BUILDING = '00809_Qpor2mEya8F'

app = Flask(__name__, static_folder=None)

_bridge = None                # UnityTelloBridge (--unity-host 줬을 때만)
_xform = None                 # SimTransform (Unity <-> 월드 좌표)


def connect_sim(host, cmd_port, local_port, state_port):
    """Unity 시뮬레이터에 붙어 드론 실시간 위치를 받는다. 실패해도 치명적이지 않다."""
    global _bridge, _xform
    from coord_transform import load_building_transform
    from unity_bridge import UnityTelloBridge

    _xform = load_building_transform(BUILDING)
    _bridge = UnityTelloBridge(unity_host=host, command_port=cmd_port,
                               local_command_port=local_port, local_state_port=state_port)
    _bridge.connect()
    st = _bridge.wait_for_state(timeout=3.0)
    if st is None:
        print(f'[web_server] 경고: {host}:{cmd_port} 에서 상태 패킷이 안 옵니다 '
              f'- Unity Play 중인지 확인. 일단 홈 위치로 대체합니다')
    else:
        print(f'[web_server] 시뮬레이터 연결됨 - unity=({st.x:.2f}, {st.y:.2f}, {st.z:.2f})')


def drone_pose():
    """드론 현재 위치를 월드(RL) 좌표로. (좌표, 출처, flying) 반환."""
    if _bridge is not None:
        st = _bridge.get_latest_state()
        if st is not None:
            p = _xform.unity_to_mosaic(_np.array([st.x, st.y, st.z], dtype=float))
            return [round(float(v), 3) for v in p], 'sim', bool(st.flying)
    return [round(v, 3) for v in rl_planner.HOME_WORLD], 'home', False


# ── API ────────────────────────────────────────────────────────────
@app.get('/api/status')
def api_status():
    return jsonify(ready=rl_planner.ready(), engine=rl_planner.engine(),
                   model=os.path.relpath(rl_planner.MODEL + '.zip', _ROOT)
                         if rl_planner.ready() else None,
                   env=dict(rl_planner.ENV_KW, max_steps=rl_planner.MAX_STEPS),
                   sim_connected=_bridge is not None,
                   camera=gesture_cam.state()['active'])


@app.get('/api/drone')
def api_drone():
    pos, source, flying = drone_pose()
    return jsonify(x=pos[0], y=pos[1], z=pos[2], source=source, flying=flying,
                   connected=_bridge is not None)


@app.get('/api/gesture')
def api_gesture():
    """현재 인식된 제스처와 rc 벡터. 카메라가 꺼져 있으면 active=False."""
    return jsonify(gesture_cam.state())


@app.get('/api/camera')
def api_camera():
    """웹캠 미리보기(MJPEG). <img src="api/camera"> 로 바로 붙는다.

    브라우저가 카메라를 직접 잡으면 파이썬 쪽 인식과 장치를 다투게 되므로,
    여기서 잡은 프레임(랜드마크·인식 결과가 그려진 것)을 넘겨준다.
    """
    if not gesture_cam.state()['active']:
        return jsonify(error='카메라가 꺼져 있습니다 (--camera 로 켜세요)'), 503

    crlf = bytes([13, 10])
    head = b'--frame' + crlf + b'Content-Type: image/jpeg' + crlf

    def gen():
        last = None
        while gesture_cam.state()['active']:
            f = gesture_cam.latest_jpeg()
            if f is not None and f is not last:
                last = f
                yield (head + b'Content-Length: ' + str(len(f)).encode()
                       + crlf + crlf + f + crlf)
            time.sleep(0.04)                      # 최대 25fps

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.post('/plan')
def api_plan():
    if not rl_planner.ready():
        return jsonify(error='RL 정책이 로드되지 않았습니다 (--no-rl 로 기동됨)'), 503
    d = request.get_json(silent=True) or {}
    # 출발점은 드론 현재 위치가 원칙. 요청에 start 가 없으면 여기서 채운다.
    start_src = 'request'
    try:
        goal = [float(v) for v in d['goal']]
        if d.get('start') is None:
            start, start_src, _ = drone_pose()
        else:
            start = [float(v) for v in d['start']]
        if len(start) != 3 or len(goal) != 3:
            raise ValueError('start/goal 은 [x, y, z] 3원소여야 합니다')
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f'잘못된 요청: {e}'), 400

    wps, info = rl_planner.plan(start, goal,
                                people=int(d.get('people', 0) or 0),
                                shield=bool(d.get('shield', False)))
    r3 = lambda seq: [[round(float(c), 3) for c in p] for p in seq]
    return jsonify(
        engine=info['engine'],
        start_source=start_src,
        success=info['success'],
        steps=info['steps'],
        bumps=info['bumps'],
        person_hits=info['person_hits'],
        dist=info['dist'],
        flown=info['raw_length_m'],
        ms=info['ms'],
        start=[round(float(v), 3) for v in info['start']],
        goal=[round(float(v), 3) for v in info['goal']],
        requested_start=[round(v, 3) for v in start],
        requested_goal=[round(v, 3) for v in goal],
        path=r3(info['raw']),        # 정책이 실제 지난 조밀 궤적 (지도에 그리는 용)
        waypoints=r3(wps),           # 시야 단축본 (시뮬레이터 비행용)
    )


# ── 정적 파일 ──────────────────────────────────────────────────────
@app.get('/')
def index():
    return redirect('/' + _INDEX.replace(' ', '%20'))


@app.get('/<path:p>')
def static_file(p):
    full = os.path.normpath(os.path.join(_WEB, p))
    if not full.startswith(_WEB) or not os.path.isfile(full):
        return jsonify(error='not found'), 404
    return send_from_directory(_WEB, p)


def main():
    ap = argparse.ArgumentParser(description='웹 관제 UI + 강화학습 경로계획 서버')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--no-rl', action='store_true',
                    help='정책을 로드하지 않고 UI 만 서빙 (프론트가 방그래프 경로로 폴백)')
    ap.add_argument('--unity-host', default=None,
                    help='Unity 시뮬레이터 IP. 주면 드론 실시간 위치를 출발점으로 쓴다 '
                         '(생략 시 시뮬레이터 스폰 지점 고정)')
    ap.add_argument('--unity-port', type=int, default=9000)
    ap.add_argument('--local-port', type=int, default=9001)
    ap.add_argument('--state-port', type=int, default=9002)
    ap.add_argument('--camera', nargs='?', type=int, const=0, default=None,
                    metavar='INDEX',
                    help='웹캠 제스처 인식을 켠다 (인덱스 생략 시 0). '
                         '웹 우하단에 미리보기가 뜬다')
    a = ap.parse_args()

    if not os.path.isdir(_WEB):
        raise SystemExit(f'웹 에셋 폴더가 없습니다: {_WEB}')
    if a.camera is not None:
        if not gesture_cam.start(a.camera):
            print('[web_server] 카메라 없이 계속합니다')
    if a.unity_host:
        try:
            connect_sim(a.unity_host, a.unity_port, a.local_port, a.state_port)
        except Exception as e:                        # 시뮬레이터는 선택 사항
            print(f'[web_server] 시뮬레이터 연결 실패({e}) - 홈 위치로 진행합니다')
    else:
        print(f'[web_server] 시뮬레이터 미연결 - 출발점은 홈 {rl_planner.HOME_WORLD} 고정 '
              f'(--unity-host 로 연결하면 실시간 위치 사용)')
    if a.no_rl:
        print('[web_server] --no-rl: 정책 없이 UI 만 서빙합니다')
    else:
        print('[web_server] 정책 로드 중… (약 6초)')
        rl_planner.load()
    print(f'[web_server] http://{a.host}:{a.port}/  - Ctrl+C 로 종료')
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == '__main__':
    main()
