# -*- coding: utf-8 -*-
"""
web_server.py — 웹 관제 UI(HAUNTED OPS) + 강화학습 경로계획 백엔드
──────────────────────────────────────────────────────────────────
(파일에 원래 있던 설명과 기능은 유지)

추가된 기능:
- POST /api/report : 원시 순찰/비행 리포트(JSON)를 받아 LLM으로 요약하고 web/uploads/reports/<id>.json 으로 저장
- GET  /api/reports : 저장된 리포트 목록 반환
- GET  /api/reports/<id> : 특정 리포트 반환

요약은 tools/report_summarizer.summarize(data) 를 사용합니다. Ollama/transformers 가 환경에 있으면 이를 사용해 요약하고, 없으면 간단한 포맷 요약을 반환합니다.
"""
import os
import sys
import time
import argparse
import threading
import uuid
import json
from datetime import datetime

import numpy as _np
from flask import Flask, Response, request, jsonify, redirect, send_from_directory

import rl_planner               # 경로 계획(정책 로드/롤아웃/단축)은 전부 여기

# repo root
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.join(_ROOT, 'simulator', 'bridge')
_WEB = os.path.join(_ROOT, 'web')
_INDEX = 'HAUNTED OPS.dc.html'

# ensure simulator bridge is importable
sys.path.insert(0, _SIM)
# ensure repo root is importable for tools/*.py
sys.path.insert(0, _ROOT)

# try import summarizer tools (may not be present in some envs)
try:
    import tools.report_summarizer as report_summarizer
except Exception:
    report_summarizer = None

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
                   sim_connected=_bridge is not None)


@app.get('/api/drone')
def api_drone():
    pos, source, flying = drone_pose()
    return jsonify(x=pos[0], y=pos[1], z=pos[2], source=source, flying=flying,
                   connected=_bridge is not None)


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


# ── 리포트 엔드포인트 (추가) ───────────────────────────────────────
# 저장 경로: web/uploads/reports/<id>.json
REPORTS_DIR = os.path.join(_WEB, 'uploads', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

@app.post('/api/report')
def api_report():
    """원시 순찰/비행 리포트(JSON)를 받아 요약한 리포트를 저장하고 반환합니다.
    요청 본문: JSON (자유 포맷 — 예: traj.json / unity_autopilot_3d_result.json 형식의 dict)
    응답: {success: bool, id: str, report: {...}}
    """
    d = request.get_json(silent=True)
    if not d:
        return jsonify(error='빈 요청입니다'), 400

    try:
        if report_summarizer is not None:
            report = report_summarizer.summarize(d)
        else:
            # 간단한 포맷 요약(폴백)
            report = _fallback_summarize(d)
    except Exception as e:
        # 예외가 나도 저장 가능한 최소 리포트로 응답
        report = {
            'id': f'report-{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}-{uuid.uuid4().hex[:6]}',
            'error': str(e),
            'raw': d,
        }

    # ensure id
    rid = report.get('id') or f'report-{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}-{uuid.uuid4().hex[:6]}'
    report['id'] = rid

    out_path = os.path.join(REPORTS_DIR, rid + '.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return jsonify(success=True, id=rid, report=report)


@app.get('/api/reports')
def api_reports_list():
    files = [f for f in os.listdir(REPORTS_DIR) if f.endswith('.json')]
    files.sort(reverse=True)
    ids = [os.path.splitext(f)[0] for f in files]
    return jsonify(reports=ids)


@app.get('/api/reports/<rid>')
def api_report_get(rid):
    path = os.path.join(REPORTS_DIR, rid + '.json')
    if not os.path.isfile(path):
        return jsonify(error='not found'), 404
    with open(path, 'r', encoding='utf-8') as f:
        r = json.load(f)
    return jsonify(r)


# ── 정적 파일 ─────────────────────────────────────────────────────
@app.get('/')
def index():
    return redirect('/' + _INDEX.replace(' ', '%20'))


@app.get('/<path:p>')
def static_file(p):
    full = os.path.normpath(os.path.join(_WEB, p))
    if not full.startswith(_WEB) or not os.path.isfile(full):
        return jsonify(error='not found'), 404
    return send_from_directory(_WEB, p)


def _fallback_summarize(d: dict) -> dict:
    """간단한 폴백 요약: 제공된 키에서 기본 메타 정보 추출해서 리포트 생성."""
    now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    rid = f'report-{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}-{uuid.uuid4().hex[:6]}'
    total_detections = d.get('total_detections') or len(d.get('detections', [])) if isinstance(d.get('detections'), list) else d.get('total_detections', 0)
    first_detection = d.get('first_detection') or (d.get('detections')[0] if d.get('detections') else None)
    flight = d.get('flight') or {}
    failures = d.get('failures') or []

    summary = {
        'id': rid,
        'mission_start': d.get('mission_start') or d.get('start_time') or None,
        'mission_end': d.get('mission_end') or None,
        'total_detections': total_detections,
        'first_detection': first_detection,
        'flight': flight,
        'failures': failures,
        'detections': d.get('detections') or [],
        'summary_text_ko': f"순찰 중 {total_detections}건의 탐지. 최초 탐지: {first_detection}.",
        'recommended_actions_ko': [],
        'raw': d,
        'generated_at': now,
    }
    return summary


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
    a = ap.parse_args()

    if not os.path.isdir(_WEB):
        raise SystemExit(f'웹 에셋 폴더가 없습니다: {_WEB}')
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
