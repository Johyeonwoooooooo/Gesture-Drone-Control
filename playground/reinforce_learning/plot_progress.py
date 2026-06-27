"""
plot_progress.py
────────────────
학습하면서 success_rate / ep_rew_mean 이 어떻게 변했는지 그래프로 본다.
(train.py 가 runs/ 에 남긴 tensorboard 로그를 읽어서 matplotlib 으로 그림)

사용:
  python plot_progress.py                # 가장 최근 학습(runs/ 최신 폴더)
  python plot_progress.py --run SAC_2    # 특정 학습 지정
  python plot_progress.py --list         # 학습 로그 목록만
저장: progress.png  (창도 같이 뜸)
"""
import os
import glob
import argparse
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(_BASE, 'runs')

# 한글 라벨 깨짐 방지 (맑은 고딕)
import matplotlib
import matplotlib.pyplot as plt
for _f in (r"C:\Windows\Fonts\malgun.ttf",):
    if os.path.exists(_f):
        matplotlib.font_manager.fontManager.addfont(_f)
        matplotlib.rcParams['font.family'] = 'Malgun Gothic'
        break
matplotlib.rcParams['axes.unicode_minus'] = False


def load_scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    out = {}
    for tag in ('rollout/success_rate', 'rollout/ep_rew_mean', 'rollout/ep_len_mean'):
        if tag in tags:
            ev = ea.Scalars(tag)
            out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None, help='runs/ 하위 폴더명 (기본: 최신)')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    runs = [d for d in glob.glob(os.path.join(_RUNS, '*')) if os.path.isdir(d)]
    if not runs:
        raise SystemExit(f"학습 로그가 없습니다: {_RUNS}")
    if args.list:
        print("학습 로그 목록 (runs/):")
        for d in sorted(runs):
            print("  ", os.path.basename(d))
        return

    run_dir = os.path.join(_RUNS, args.run) if args.run else max(runs, key=os.path.getmtime)
    print(f"[그래프] 학습 로그: {os.path.basename(run_dir)}")
    sc = load_scalars(run_dir)
    if 'rollout/success_rate' not in sc:
        raise SystemExit("이 로그에 success_rate 가 없습니다. (--list 로 다른 run 확인)")

    fig, ax1 = plt.subplots(figsize=(11, 6))
    s_step, s_val = sc['rollout/success_rate']
    ax1.plot(s_step, s_val, color='tab:green', lw=2, label='성공률(success_rate)')
    ax1.set_xlabel('학습 스텝 (timesteps)')
    ax1.set_ylabel('성공률', color='tab:green')
    ax1.set_ylim(-0.02, 1.02)
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax1.grid(alpha=0.3)

    if 'rollout/ep_rew_mean' in sc:                    # 보상은 오른쪽 축에 겹쳐 그림
        ax2 = ax1.twinx()
        r_step, r_val = sc['rollout/ep_rew_mean']
        ax2.plot(r_step, r_val, color='tab:blue', lw=1.2, alpha=0.7,
                 label='평균보상(ep_rew_mean)')
        ax2.set_ylabel('평균 에피소드 보상', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

    peak = float(np.max(s_val)); peak_at = int(s_step[int(np.argmax(s_val))])
    plt.title(f"학습 진행 — {os.path.basename(run_dir)}  "
              f"(최고 성공률 {peak:.2f} @ {peak_at:,} step, 최종 {s_val[-1]:.2f})")
    fig.tight_layout()

    out = os.path.join(_BASE, 'progress.png')
    fig.savefig(out, dpi=120)
    print(f"[그래프] 저장: {out}")
    plt.show()


if __name__ == '__main__':
    main()
