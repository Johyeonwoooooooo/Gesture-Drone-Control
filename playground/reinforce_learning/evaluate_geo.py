"""
evaluate_geo.py — v2(geo) 모델 평가·시연: 전체집, 층 넘기 포함
──────────────────────────────────────────────────────────────
1) 점 2개 찍어 비행 + 3D 시각화 (층이 달라도 됨 — 계단으로 넘어가는지 보기):
     python evaluate_geo.py --pick
     python evaluate_geo.py --start 6 6 4.2 --goal 2 0 -2.0
2) 무작위 배치 평가 (같은 층 / 층 넘기 성공률 분리 리포트):
     python evaluate_geo.py --random 60
3) 특권 정보 경계 증명 (priv 를 0으로 밀어도 행동이 같아야 함 — actor 는 센서만 봄):
     python evaluate_geo.py --random 20 --prove-boundary

기본 모델은 model_geo_best (없으면 model_geo).
"""
import os
import argparse
import numpy as np

from geo_env import DroneGeoEnv
import asym_policy  # noqa: F401  SAC.load 가 AsymSACPolicy 를 찾을 수 있게
from evaluate import pick_two_points, visualize

_BASE = os.path.dirname(os.path.abspath(__file__))


def fly(env, model, start=None, goal=None, zero_priv=False):
    opts = {}
    if start is not None:
        opts['start'] = np.asarray(start, float)
    if goal is not None:
        opts['goal'] = np.asarray(goal, float)
    obs, _ = env.reset(options=opts if opts else None)
    term = trunc = False
    while not (term or trunc):
        if zero_priv and isinstance(obs, dict):
            obs = {'sensor': obs['sensor'], 'priv': np.zeros_like(obs['priv'])}
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
    return info


def main():
    ap = argparse.ArgumentParser(description='v2(geo) 평가: 전체집, 층 넘기 포함')
    ap.add_argument('--model', default=None, help='모델 경로(확장자 제외). 기본 model_geo_best')
    ap.add_argument('--pick', action='store_true', help='마우스로 점 2개 찍기')
    ap.add_argument('--start', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--goal',  type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--random', type=int, default=0, help='무작위 N 에피소드 배치 평가')
    ap.add_argument('--prove-boundary', action='store_true',
                    help='priv 관측을 0으로 바꿔도 결과가 같은지 확인(경계 증명)')
    ap.add_argument('--goal-radius', type=float, default=0.5)
    ap.add_argument('--max-steps', type=int, default=600, help='시연용 스텝 여유(기본 600)')
    ap.add_argument('--ray-max', type=float, default=2.0,
                    help='거리센서 사거리(m). 학습 때 쓴 값과 반드시 동일하게')
    ap.add_argument('--rays', choices=['axes14', 'horiz14'], default='axes14',
                    help='레이 배치. 학습 때 쓴 값과 반드시 동일하게')
    ap.add_argument('--subgoal', type=float, default=None,
                    help='계층 모드 carrot 거리(m). 학습 때 쓴 값과 반드시 동일하게')
    ap.add_argument('--bump', type=float, default=None,
                    help='범퍼 물리 페널티. 학습 때 쓴 값과 반드시 동일하게')
    ap.add_argument('--clearance', type=float, default=None,
                    help='여유반경(m). 학습 때 쓴 값과 반드시 동일하게')
    ap.add_argument('--guide-clearance', type=float, default=0.20,
                    help='안내(측지 그래프) 최소 여유(m). 학습 때 쓴 값과 동일하게'
                         ' (구버전 키홀 재현은 0.12)')
    args = ap.parse_args()

    model_path = args.model or os.path.join(_BASE, 'model_geo_best')
    if not os.path.exists(model_path + '.zip'):
        alt = os.path.join(_BASE, 'model_geo')
        if args.model is None and os.path.exists(alt + '.zip'):
            model_path = alt
        else:
            raise SystemExit(f"모델이 없습니다: {model_path}.zip  먼저 train_geo.py 로 학습하세요.")

    from stable_baselines3 import SAC
    model = SAC.load(model_path)
    priv = isinstance(model.observation_space, __import__('gymnasium').spaces.Dict)
    _ckw = {} if args.clearance is None else {'clearance': args.clearance}
    env = DroneGeoEnv(curriculum=False, priv_obs=priv, ray_max=args.ray_max,
                      ray_layout=args.rays, subgoal_dist=args.subgoal,
                      bump_penalty=args.bump, guide_clearance=args.guide_clearance,
                      goal_radius=args.goal_radius, max_steps=args.max_steps, **_ckw)
    print(f"[평가] 모델 {os.path.basename(model_path)}.zip  (priv_obs={priv})")

    if args.random > 0:                                    # ── 배치 평가 ──
        cats = {'같은 층': [0, 0], '층 넘기': [0, 0]}      # [성공, 시도]
        steps_ok = []
        for i in range(args.random):
            env.reset(seed=i)                               # 에피소드 고정(재현 가능)
            start, goal = env.pos.copy(), env.goal.copy()
            info = fly(env, model, start=start, goal=goal,
                       zero_priv=args.prove_boundary)
            cat = '층 넘기' if abs(goal[2] - start[2]) > 1.5 else '같은 층'
            cats[cat][1] += 1
            cats[cat][0] += int(info['is_success'])
            if info['is_success']:
                steps_ok.append(env.steps)
            print(f"  ep{i:03d} {cat} |Δz|={abs(goal[2]-start[2]):.1f}m "
                  f"→ {'성공' if info['is_success'] else '실패'} ({env.steps}스텝)")
        print("\n[결과]")
        tot_s = tot_n = 0
        for k, (s, n) in cats.items():
            if n:
                print(f"  {k}: {s}/{n} = {s/n:.2f}")
            tot_s += s; tot_n += n
        print(f"  전체  : {tot_s}/{tot_n} = {tot_s/max(tot_n,1):.2f}"
              + (f" | 성공 평균 {np.mean(steps_ok):.0f}스텝" if steps_ok else ""))
        if args.prove_boundary:
            print("  (priv=0 강제 상태의 결과. 평소와 같으면 actor 가 특권 정보를"
                  " 안 쓴다는 증명)")
        return

    # ── 단일 비행 + 3D 시각화 ──
    if args.pick or (args.start is None or args.goal is None):
        print("\n[팁] 층이 다른 두 점을 찍으면 계단으로 넘어가는지 볼 수 있어.")
        start, goal = pick_two_points(env.coord)
    else:
        start, goal = np.array(args.start), np.array(args.goal)
    print(f"  시작 ≈ {np.round(start, 2)}   도착 ≈ {np.round(goal, 2)}")

    info = fly(env, model, start=start, goal=goal, zero_priv=args.prove_boundary)
    flown = float(np.linalg.norm(np.diff(np.array(env.path), axis=0), axis=1).sum()) \
        if len(env.path) > 1 else 0.0
    print(f"  결과: {'도착 성공' if info['is_success'] else '실패'} | {env.steps}스텝 | "
          f"비행거리 {flown:.2f}m | 남은거리 {info['dist']:.2f}m")
    visualize(env.coord, env.path, np.array(env.path[0]), env.goal, info['is_success'])


if __name__ == '__main__':
    main()
