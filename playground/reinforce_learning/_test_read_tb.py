# -*- coding: utf-8 -*-
"""Dump latest tensorboard scalars from a run dir (ASCII)."""
import sys, glob, os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1] if len(sys.argv) > 1 else "runs/SAC_42"
ef = sorted(glob.glob(os.path.join(run, "events.out.tfevents.*")))[-1]
ea = EventAccumulator(ef, size_guidance={'scalars': 0})
ea.Reload()
tags = ea.Tags()['scalars']

def last(tag, n=1):
    if tag not in tags:
        return None
    ev = ea.Scalars(tag)
    return ev[-n:]

want = ['time/total_timesteps', 'rollout/d_max', 'goal/d_max_m',
        'rollout/success_rate', 'eval/success_rate', 'rollout/ep_rew_mean',
        'eval/mean_reward', 'train/ent_coef', 'train/critic_loss',
        'train/actor_loss', 'eval/mean_ep_length']
print(f"[tb] file={os.path.basename(ef)}")
print(f"[tb] available tags: {tags}\n")
for t in want:
    ev = last(t, 1)
    if ev:
        print(f"  {t:28s} step={ev[-1].step:>9,}  val={ev[-1].value:.4f}")

# trend for success_rate + d_max (last 8 points)
for t in ['eval/success_rate', 'rollout/success_rate', 'rollout/d_max']:
    ev = last(t, 8)
    if ev:
        vals = ", ".join(f"{e.value:.2f}" for e in ev)
        print(f"\n  {t} (최근8): {vals}")
