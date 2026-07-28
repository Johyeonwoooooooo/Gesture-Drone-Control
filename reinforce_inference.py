# -*- coding: utf-8 -*-
"""
reinforce_inference.py — 학습된 SAC 정책 단독 추론(비행 데모·평가) 진입점
────────────────────────────────────────────────────────────────────
git clone 직후 바로 실행 가능:

    pip install -r requirements.txt
    python reinforce_inference.py                    # 무작위 5 에피소드 헤드리스 평가
    python reinforce_inference.py --random 100       # 대규모 평가(같은 층/층 넘기 분리)
    python reinforce_inference.py --pick             # 3D에서 점 2개 찍어 비행 + 시각화
    python reinforce_inference.py --start 6 6 4.2 --goal 2 0 -2.0 --viz
    python reinforce_inference.py --random 20 --people 2           # 동적 장애물(사람)
    python reinforce_inference.py --random 20 --people 2 --shield  # + 런타임 안전막

필요한 데이터는 전부 저장소에 포함돼 있어 추가 다운로드/스캔이 필요 없다:
  - playground/reinforce_learning/model_geo_best.zip      모델 파라미터(SAC)
  - playground/reinforce_learning/*.npz                   장애물·측지 그래프·계단 캐시
  - playground/demo_house/rooms_graph.json                방 그래프

정책은 순수 reactive(레이 센서 + carrot 서브골)로 비행하며, 측지 필드는 서브골
안내에만 쓰인다. --viz / --pick 은 open3d 창을 띄우므로 GUI 환경에서만 사용.
"""
import os
import sys
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_RL = os.path.join(_ROOT, 'playground', 'reinforce_learning')
sys.path.insert(0, _RL)

# 학습 당시 환경 설정 — 모델과 반드시 일치해야 함(바꾸면 성능 붕괴)
ENV_KW = dict(curriculum=False, ray_max=4.0, ray_layout='horiz14',
              subgoal_dist=2.5, bump_penalty=2.0, clearance=0.12)

DEFAULT_MODEL = os.path.join(_RL, 'model_geo_best')


def rollout(env, model, obs):
    """현재 obs 에서 에피소드 끝까지 결정론 비행. 마지막 info 반환."""
    term = trunc = False
    info = {}
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
    return info


def main():
    ap = argparse.ArgumentParser(description='학습된 SAC 정책으로 실내 자율비행 추론')
    ap.add_argument('--model', default=None, help='모델 경로(.zip 생략 가능). '
                    '기본: playground/reinforce_learning/model_geo_best')
    ap.add_argument('--random', type=int, default=5, help='무작위 N 에피소드 평가(기본 5)')
    ap.add_argument('--pick', action='store_true', help='3D 뷰어에서 시작/도착 점 찍기')
    ap.add_argument('--start', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--goal', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--viz', action='store_true', help='비행 후 open3d 3D 시각화')
    ap.add_argument('--people', type=int, default=0, help='동적 장애물(순찰 보행자) 수')
    ap.add_argument('--shield', action='store_true', help='런타임 안전막(사람 근접 시 개입)')
    ap.add_argument('--max-steps', type=int, default=700)
    ap.add_argument('--seed', type=int, default=0, help='무작위 평가 시드 베이스')
    a = ap.parse_args()

    import gymnasium
    import asym_policy  # noqa: F401  SAC.load 가 AsymSACPolicy 를 찾을 수 있게
    from stable_baselines3 import SAC
    from geo_env import DroneGeoEnv

    model_path = a.model or DEFAULT_MODEL
    if model_path.endswith('.zip'):
        model_path = model_path[:-4]
    if not os.path.exists(model_path + '.zip'):
        raise SystemExit(f'모델이 없습니다: {model_path}.zip')
    print(f'[모델] {os.path.relpath(model_path + ".zip", _ROOT)}')
    model = SAC.load(model_path, device='auto')
    priv = isinstance(model.observation_space, gymnasium.spaces.Dict)

    env = DroneGeoEnv(priv_obs=priv, max_steps=a.max_steps,
                      people=a.people, shield=a.shield, **ENV_KW)

    # ── 단일 비행 (--pick 또는 --start/--goal) ──────────────────
    if a.pick or (a.start is not None and a.goal is not None):
        if a.pick:
            from evaluate import pick_two_points
            print('[팁] 층이 다른 두 점을 찍으면 계단으로 넘어가는지 볼 수 있다.')
            start, goal = pick_two_points(env.coord)
        else:
            start, goal = np.asarray(a.start, float), np.asarray(a.goal, float)
        print(f'  시작 = {np.round(start, 2)}   도착 = {np.round(goal, 2)}')
        obs, _ = env.reset(options={'start': start, 'goal': goal})
        info = rollout(env, model, obs)
        flown = float(np.linalg.norm(np.diff(np.array(env.path), axis=0), axis=1).sum()) \
            if len(env.path) > 1 else 0.0
        print(f'  결과: {"도착 성공" if info["is_success"] else "실패"} | {env.steps}스텝 | '
              f'비행거리 {flown:.2f}m | 남은거리 {info["dist"]:.2f}m | 범프 {info["bumps"]}')
        out = os.path.join(_ROOT, 'flight_path.npy')
        np.save(out, np.array(env.path))
        print(f'  궤적 저장: {out}')
        if a.viz or a.pick:
            from evaluate import visualize
            visualize(env.coord, env.path, np.array(env.path[0]), env.goal,
                      info['is_success'])
        return

    # ── 무작위 배치 평가 ────────────────────────────────────────
    n = max(1, a.random)
    print(f'[평가] 무작위 {n} 에피소드 (people={a.people}, shield={a.shield})')
    cats = {'같은 층': [0, 0], '층 넘기': [0, 0]}
    steps_ok, contact_free = [], 0
    for i in range(n):
        obs, _ = env.reset(seed=a.seed + i)
        start, goal = env.pos.copy(), env.goal.copy()
        info = rollout(env, model, obs)
        cat = '층 넘기' if abs(goal[2] - start[2]) > 1.5 else '같은 층'
        cats[cat][1] += 1
        cats[cat][0] += int(info['is_success'])
        phits = int(info.get('person_hits', 0))
        if info['is_success']:
            steps_ok.append(env.steps)
            contact_free += int(phits == 0)
        extra = f' | 사람접촉 {phits}' if a.people else ''
        print(f'  ep{i:03d} {cat} |Δz|={abs(goal[2]-start[2]):.1f}m → '
              f'{"성공" if info["is_success"] else "실패"} ({env.steps}스텝{extra})')
    print('\n[결과]')
    tot_s = tot_n = 0
    for k, (s, m) in cats.items():
        if m:
            print(f'  {k}: {s}/{m} = {s/m:.2f}')
        tot_s += s
        tot_n += m
    line = f'  전체  : {tot_s}/{tot_n} = {tot_s/max(tot_n,1):.2f}'
    if steps_ok:
        line += f' | 성공 평균 {np.mean(steps_ok):.0f}스텝'
    if a.people:
        line += f' | 비접촉 성공 {contact_free}/{tot_n} = {contact_free/max(tot_n,1):.2f}'
    print(line)


if __name__ == '__main__':
    main()
