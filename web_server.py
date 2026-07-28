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

API
  GET  /api/status  →  {ready, engine, model, env}
  POST /plan        →  {start,goal,path,success,steps,bumps,dist,flown,ms,engine}
       body: {"start":[x,y,z], "goal":[x,y,z], "people":0, "shield":false}

주의: 모델 로드 + 환경 생성은 기동 시 1회(약 6초)만 하고 상주시킨다.
한 번의 경로 계획은 0.05~0.2초(최악 700스텝 ≈ 3.5초). 환경 객체는
재진입 불가라 Lock 으로 직렬화한다.
"""
import os
import sys
import time
import argparse
import threading

from flask import Flask, request, jsonify, redirect, send_from_directory

_ROOT = os.path.dirname(os.path.abspath(__file__))
_RL = os.path.join(_ROOT, 'playground', 'reinforce_learning')
_WEB = os.path.join(_ROOT, 'web')
_INDEX = 'HAUNTED OPS.dc.html'

sys.path.insert(0, _RL)

# 학습 당시 환경 설정 — reinforce_inference.py 와 동일해야 한다(바꾸면 성능 붕괴).
# ★ clearance 는 항상 명시할 것: 기본값(0.235)으로 env 를 만들면 0.12 용
#   geo_graph_cache 가 덮여서 다음 실행이 캐시를 1분간 재생성한다.
ENV_KW = dict(curriculum=False, ray_max=4.0, ray_layout='horiz14',
              subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12)
MAX_STEPS = 700
MODEL = os.path.join(_RL, 'model_geo_best')

app = Flask(__name__, static_folder=None)

_lock = threading.Lock()      # env 는 재진입 불가 → 요청 직렬화
_model = None
_envs = {}                    # (people, shield) -> DroneGeoEnv
_priv = False
_engine = 'fallback (RL 비활성)'
_np = None


def load_policy():
    """모델 + 환경을 1회 로드한다. 실패하면 예외를 올린다."""
    global _model, _priv, _engine, _np
    import numpy as np
    import gymnasium
    import asym_policy                       # noqa: F401  SAC.load 가 정책 클래스를 찾게
    from stable_baselines3 import SAC

    _np = np
    t0 = time.time()
    if not os.path.exists(MODEL + '.zip'):
        raise SystemExit(f'모델이 없습니다: {MODEL}.zip')
    _model = SAC.load(MODEL, device='auto')
    _priv = isinstance(_model.observation_space, gymnasium.spaces.Dict)
    get_env(0, False)                        # 기본 환경을 미리 생성(캐시 워밍)
    _engine = f'SAC · {os.path.basename(MODEL)}'
    print(f'[web_server] 정책 준비 완료 ({time.time() - t0:.1f}s) — {_engine}')


def get_env(people, shield):
    """(사람 수, 안전막) 조합별 환경을 지연 생성해 재사용한다."""
    key = (int(people), bool(shield))
    env = _envs.get(key)
    if env is None:
        from geo_env import DroneGeoEnv
        env = DroneGeoEnv(priv_obs=_priv, max_steps=MAX_STEPS,
                          people=key[0], shield=key[1], **ENV_KW)
        _envs[key] = env
    return env


def rollout(env, obs):
    """결정론 롤아웃. (마지막 info, 궤적 ndarray) 반환."""
    term = trunc = False
    info = {}
    while not (term or trunc):
        action, _ = _model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
    return info, _np.array(env.path)


# ── API ────────────────────────────────────────────────────────────
@app.get('/api/status')
def api_status():
    return jsonify(ready=_model is not None, engine=_engine,
                   model=os.path.relpath(MODEL + '.zip', _ROOT) if _model else None,
                   env=dict(ENV_KW, max_steps=MAX_STEPS))


@app.post('/plan')
def api_plan():
    if _model is None:
        return jsonify(error='RL 정책이 로드되지 않았습니다 (--no-rl 로 기동됨)'), 503
    d = request.get_json(silent=True) or {}
    try:
        start = [float(v) for v in d['start']]
        goal = [float(v) for v in d['goal']]
        if len(start) != 3 or len(goal) != 3:
            raise ValueError('start/goal 은 [x, y, z] 3원소여야 합니다')
    except (KeyError, TypeError, ValueError) as e:
        return jsonify(error=f'잘못된 요청: {e}'), 400
    people = int(d.get('people', 0) or 0)
    shield = bool(d.get('shield', False))

    t0 = time.time()
    with _lock:
        env = get_env(people, shield)
        # 벽 위를 찍어도 env 가 가장 가까운 도달 가능 셀로 스냅해 준다.
        obs, _ = env.reset(options={'start': _np.asarray(start, float),
                                    'goal': _np.asarray(goal, float)})
        snapped_start = env.pos.copy()
        snapped_goal = env.goal.copy()
        info, path = rollout(env, obs)
        steps = int(env.steps)
    ms = (time.time() - t0) * 1000.0

    flown = float(_np.linalg.norm(_np.diff(path, axis=0), axis=1).sum()) if len(path) > 1 else 0.0
    return jsonify(
        engine=_engine,
        success=bool(info['is_success']),
        steps=steps,
        bumps=int(info.get('bumps', 0)),
        person_hits=int(info.get('person_hits', 0)),
        dist=float(info['dist']),
        flown=flown,
        ms=ms,
        start=[round(v, 3) for v in snapped_start.tolist()],
        goal=[round(v, 3) for v in snapped_goal.tolist()],
        requested_start=[round(v, 3) for v in start],
        requested_goal=[round(v, 3) for v in goal],
        path=[[round(float(c), 3) for c in p] for p in path],
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
    a = ap.parse_args()

    if not os.path.isdir(_WEB):
        raise SystemExit(f'웹 에셋 폴더가 없습니다: {_WEB}')
    if a.no_rl:
        print('[web_server] --no-rl: 정책 없이 UI 만 서빙합니다')
    else:
        print('[web_server] 정책 로드 중… (약 6초)')
        load_policy()
    print(f'[web_server] http://{a.host}:{a.port}/  — Ctrl+C 로 종료')
    app.run(host=a.host, port=a.port, threaded=True)


if __name__ == '__main__':
    main()
