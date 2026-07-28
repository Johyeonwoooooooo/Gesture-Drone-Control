"""
fly_pick.py — 3D에서 점 2개 찍으면 강화학습 정책이 비행해 경로를 보여준다
────────────────────────────────────────────────────────────────
현재 최종 모델(model_geo_best)의 학습 설정(ray 4m/horiz14, subgoal 2.5,
bump 2, clearance 0.12, guide 0.2)이 전부 기본값으로 박혀 있어 그냥 실행하면 된다.

  python fly_pick.py --demo                   # ★ 1층 아무데나 → 3층 아무데나 자동 비행
  python fly_pick.py --demo --loop            # 위를 계속 반복 (창 닫으면 다음 비행)
  python fly_pick.py                          # 3D 창에서 Shift+클릭 2번 (시작→도착)
  python fly_pick.py --loop                   # 연속 데모: 비행 후 다시 점 찍기 반복
  python fly_pick.py --start 9.6 1.9 4.4 --goal 7 1.6 -1.5   # 좌표 직접 지정
  python fly_pick.py --model model_geo        # 다른 모델로

조작: Shift+왼클릭으로 점 2개(시작, 도착) 찍고 Q로 창 닫기.
벽/바닥 위를 찍어도 가장 가까운 '실내 비행가능 지점'으로 자동 스냅된다.
비행 궤적은 flight_path.npy 로 저장(Nx3, m) — 다른 스크립트에서 재사용 가능.
"""
import os
import argparse
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description='점 2개 찍기 → RL 비행 → 3D 경로')
    ap.add_argument('--model', default=None, help='모델(.zip 제외). 기본 model_geo_best')
    ap.add_argument('--start', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--goal',  type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--demo', action='store_true',
                    help='점 안 찍고 1층 아무데나 -> 3층 아무데나 무작위 자동 비행')
    ap.add_argument('--loop', action='store_true', help='비행 후 다시 점 찍기 반복(데모용)')
    ap.add_argument('--goal-radius', type=float, default=0.5)
    ap.add_argument('--max-steps', type=int, default=700)
    ap.add_argument('--out', default=os.path.join(_BASE, 'flight_path.npy'),
                    help='비행 궤적 저장 경로(.npy)')
    ap.add_argument('--no-viz', action='store_true',
                    help='3D 창 없이 비행+궤적 저장만 (스크립트/파이프라인용)')
    args = ap.parse_args()

    model_path = args.model or 'model_geo_best'
    model_path = os.path.join(_BASE, model_path)
    if not os.path.exists(model_path + '.zip'):
        alt = os.path.join(_BASE, 'model_geo')
        if args.model is None and os.path.exists(alt + '.zip'):
            model_path = alt
        else:
            raise SystemExit(f"모델이 없습니다: {model_path}.zip")

    from stable_baselines3 import SAC
    import asym_policy  # noqa: F401  (AsymSACPolicy 로드용)
    from geo_env import DroneGeoEnv
    from evaluate import pick_two_points, visualize

    model = SAC.load(model_path)
    priv = isinstance(model.observation_space, __import__('gymnasium').spaces.Dict)
    # ★ 학습 설정과 동일해야 하는 값들 — 현재 모델 기준으로 고정
    env = DroneGeoEnv(curriculum=False, priv_obs=priv,
                      ray_max=4.0, ray_layout='horiz14', subgoal_dist=2.5,
                      bump_penalty=2.0, clearance=0.12,
                      goal_radius=args.goal_radius, max_steps=args.max_steps)
    print(f"[fly_pick] 모델 {os.path.basename(model_path)}.zip 준비 완료")

    def one_flight(start, goal):
        obs, _ = env.reset(options={'start': np.asarray(start, float),
                                    'goal': np.asarray(goal, float)})
        s, g = env.pos.copy(), env.goal.copy()          # 스냅 후 실제 시작/목표
        print(f"  시작 {np.round(s, 2)} -> 도착 {np.round(g, 2)}"
              f" | 측지거리 {env._phi(s):.1f} m")
        term = trunc = False
        while not (term or trunc):
            act, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
        path = np.asarray(env.path)
        flown = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) \
            if len(path) > 1 else 0.0
        ok = bool(info['is_success'])
        print(f"  결과: {'도착 성공' if ok else '실패'} | {env.steps}스텝 | "
              f"비행 {flown:.1f} m | 범프 {info['bumps']} | 남은거리 {info['dist']:.2f} m")
        np.save(args.out, path)
        print(f"  궤적 저장: {os.path.basename(args.out)} ({len(path)}점)")
        if not args.no_viz:
            visualize(env.coord, path, s, g, ok)

    if args.start is not None and args.goal is not None:
        one_flight(args.start, args.goal)
        return

    if args.demo:
        # 층별 z 대역 (이 집 기준): 1층 z<-0.5 / 2층 -0.5~2.5 / 3층 z>2.5.
        # 그래프 주성분(실내 비행가능 셀)에서만 뽑으므로 벽 속/막힌 곳은 안 나온다.
        rng = np.random.default_rng()
        cells = env.node_pos[env.main_nodes]
        floor1 = cells[cells[:, 2] < -0.5]
        floor3 = cells[cells[:, 2] > 2.5]
        while True:
            start = floor1[rng.integers(len(floor1))]
            goal = floor3[rng.integers(len(floor3))]
            print(f"\n[demo] 1층 {np.round(start, 1)} -> 3층 {np.round(goal, 1)}")
            one_flight(start, goal)
            if not args.loop:
                break
        return

    while True:
        start, goal = pick_two_points(env.coord)
        one_flight(start, goal)
        if not args.loop:
            break
        print("\n[loop] 다음 비행 - 다시 점 2개를 찍으세요 (창 닫으면 종료: Ctrl+C)")


if __name__ == '__main__':
    main()
