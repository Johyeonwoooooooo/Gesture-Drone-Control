"""
evaluate.py
───────────
학습된 정책으로 '점 2개(시작/도착)' 사이를 비행시키고 3D로 결과를 본다.

두 가지 방법으로 시작/도착점 지정:
  1) 마우스로 찍기(기본):  python evaluate.py --pick
       open3d 창에서 [Shift+왼클릭] 으로 점 2개(시작→도착) 찍고 → 창 닫기(Q)
  2) 좌표 직접:            python evaluate.py --start 6 6 4.2 --goal 2 0 1.2

학습 때와 달리 여기서는 네가 고른 점을 시작/도착으로 쓴다(막힌 점이면 근처 빈 곳으로 자동 보정).
RRT* 결과(hierarchical_plan.py)와 비교해보면 RL 이 무엇을 배웠는지 감이 온다.
"""
import os
import argparse
import numpy as np

from drone_env import DroneHouseEnv

_BASE = os.path.dirname(os.path.abspath(__file__))


def pick_two_points(coord):
    """open3d 편집창에서 Shift+클릭으로 점 2개를 찍어 좌표로 반환."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.paint_uniform_color([0.6, 0.6, 0.62])
    print("\n[점 찍기] Shift+왼클릭으로 시작점 → 도착점 순서로 2개 찍고 창을 닫으세요(Q).")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="시작/도착 점 2개 찍기 (Shift+클릭)", width=1400, height=900)
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 2.5
    vis.run()
    vis.destroy_window()
    idx = vis.get_picked_points()
    if len(idx) < 2:
        raise SystemExit("점을 2개 찍어야 합니다.")
    return coord[idx[0]], coord[idx[1]]


def _sphere(c, r, col):
    import open3d as o3d
    m = o3d.geometry.TriangleMesh.create_sphere(radius=r, resolution=12)
    m.translate(np.asarray(c, float)); m.paint_uniform_color(col); m.compute_vertex_normals()
    return m


def visualize(coord, path, start, goal, success):
    import open3d as o3d
    geoms = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.paint_uniform_color([0.55, 0.55, 0.58])
    geoms.append(pcd)

    path = np.asarray(path)
    if len(path) >= 2:
        lines = [[i, i + 1] for i in range(len(path) - 1)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(path),
            lines=o3d.utility.Vector2iVector(np.asarray(lines)))
        col = [0.1, 0.9, 0.1] if success else [1.0, 0.3, 0.0]
        ls.colors = o3d.utility.Vector3dVector(np.tile(col, (len(lines), 1)))
        geoms.append(ls)

    geoms.append(_sphere(start, 0.24, [0.1, 0.9, 0.1]))   # 시작 초록
    geoms.append(_sphere(goal,  0.24, [0.9, 0.1, 0.1]))   # 도착 빨강

    tag = "도착 성공" if success else "실패(충돌/시간초과)"
    print(f"\n[3D] {tag} | 경로 {len(path)}점 (초록구=시작, 빨강구=도착, 선=RL 비행궤적)")
    o3d.visualization.draw_geometries(geoms, window_name=f"RL 비행 결과 — {tag}",
                                      width=1400, height=900)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(_BASE, 'model'), help='모델 경로(확장자 제외)')
    ap.add_argument('--algo', choices=['sac', 'ppo'], default='sac')
    ap.add_argument('--pick', action='store_true', help='마우스로 점 2개 찍기')
    ap.add_argument('--start', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--goal',  type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    ap.add_argument('--stochastic', action='store_true', help='결정론적 대신 확률적 행동')
    ap.add_argument('--goal-radius', type=float, default=0.6,
                    help='도착 판정 반경(m). easy 학습값과 맞춤(기본 0.6)')
    args = ap.parse_args()

    if not os.path.exists(args.model + '.zip'):
        raise SystemExit(f"모델이 없습니다: {args.model}.zip  먼저 train.py 로 학습하세요.")

    from stable_baselines3 import SAC, PPO
    Algo = SAC if args.algo == 'sac' else PPO

    env = DroneHouseEnv(goal_radius=args.goal_radius)
    model = Algo.load(args.model)

    if args.pick or (args.start is None or args.goal is None):
        print("\n[팁] easy 로 학습한 모델은 '가까운(같은 방, ~4m)' 목표만 배웠어.")
        print("     데모는 시작·도착을 너무 멀지 않게(같은 방/가까이) 찍어야 잘 날아가.")
        start, goal = pick_two_points(env.coord)
    else:
        start, goal = np.array(args.start), np.array(args.goal)
    print(f"  시작 ≈ {np.round(start, 2)}   도착 ≈ {np.round(goal, 2)}")

    obs, _ = env.reset(options={'start': start, 'goal': goal})
    term = trunc = False
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=not args.stochastic)
        obs, _, term, trunc, info = env.step(action)

    flown = float(np.linalg.norm(np.diff(np.array(env.path), axis=0), axis=1).sum()) \
        if len(env.path) > 1 else 0.0
    print(f"  결과: {'도착 성공' if info['is_success'] else '실패'} | "
          f"{env.steps}스텝 | 비행거리 {flown:.2f}m | 남은거리 {info['dist']:.2f}m")

    visualize(env.coord, env.path, np.array(env.path[0]), env.goal, info["is_success"])


if __name__ == '__main__':
    main()
