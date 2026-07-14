"""
plot_progress_tqc.py
────────────────────
TQC 학습 그래프. runs_tqc/ 로그를 읽는다.

사용:
  /opt/conda/bin/python plot_progress_tqc.py          # 최신 학습
  /opt/conda/bin/python plot_progress_tqc.py --list   # 목록 확인
저장: progress_tqc.png
"""
import os
import glob
import argparse
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_RUNS = os.path.join(_BASE, 'runs_tqc')

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['axes.unicode_minus'] = False


def load_scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(run_dir, size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    out = {}
    for tag in ('rollout/success_rate', 'rollout/ep_rew_mean', 'eval/success_rate'):
        if tag in tags:
            ev = ea.Scalars(tag)
            out[tag] = (np.array([e.step for e in ev]), np.array([e.value for e in ev]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run',  default=None)
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    runs = [d for d in glob.glob(os.path.join(_RUNS, '*')) if os.path.isdir(d)]
    if not runs:
        raise SystemExit(f"학습 로그가 없습니다: {_RUNS}")
    if args.list:
        for d in sorted(runs):
            print(os.path.basename(d))
        return

    run_dir = os.path.join(_RUNS, args.run) if args.run else max(runs, key=os.path.getmtime)
    print(f"[그래프] {os.path.basename(run_dir)}")
    sc = load_scalars(run_dir)

    # eval/success_rate 우선, 없으면 rollout/success_rate
    sr_key = 'eval/success_rate' if 'eval/success_rate' in sc else 'rollout/success_rate'
    if sr_key not in sc:
        raise SystemExit("success_rate 데이터가 없습니다.")

    fig, ax1 = plt.subplots(figsize=(11, 6))
    s_step, s_val = sc[sr_key]
    ax1.plot(s_step, s_val, color='tab:green', lw=2, label='Success Rate')
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Success Rate', color='tab:green')
    ax1.set_ylim(-0.02, 1.02)
    ax1.tick_params(axis='y', labelcolor='tab:green')
    ax1.grid(alpha=0.3)

    if 'rollout/ep_rew_mean' in sc:
        ax2 = ax1.twinx()
        r_step, r_val = sc['rollout/ep_rew_mean']
        ax2.plot(r_step, r_val, color='tab:blue', lw=1.2, alpha=0.7, label='Mean Reward')
        ax2.set_ylabel('Mean Episode Reward', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

    peak = float(np.max(s_val))
    peak_at = int(s_step[int(np.argmax(s_val))])
    plt.title(f"TQC Training — {os.path.basename(run_dir)}\n"
              f"Best Success Rate: {peak:.2f} @ {peak_at:,} steps  /  Final: {s_val[-1]:.2f}")
    fig.tight_layout()

    out = os.path.join(_BASE, 'progress_tqc.png')
    fig.savefig(out, dpi=120)
    print(f"[그래프] 저장: {out}")
    plt.show()


if __name__ == '__main__':
    main()
