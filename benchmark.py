"""
benchmark.py
============
Run *N* repetitions of both sequential and parallel planning on the same
task set, collect timing statistics, and produce publication-quality charts.

Charts saved
------------
* ``benchmark_<timestamp>.png``  — main 2×3 panel (wall time, sum time,
  speedup, per-room breakdown, path length, expanded nodes)
* ``benchmark_per_room_<timestamp>.png``  — per-room bar chart (optional)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from config import PALETTE, PLOT_DPI
from parallel_planner import aggregate_stats, build_tasks, run_parallel, run_sequential


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    tasks: list[tuple],
    room_sequence: list[str],
    n_runs: int = 5,
    n_workers: int | None = None,
) -> dict:
    """
    Execute *n_runs* rounds of sequential + parallel planning.

    Returns
    -------
    A dict with keys ``"sequential"`` and ``"parallel"``, each containing
    a list of per-run aggregate-stat dicts, plus raw per-room results for
    the last run.
    """
    seq_runs: list[dict] = []
    par_runs: list[dict] = []
    last_seq_results: list[dict] = []
    last_par_results: list[dict] = []

    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs} …", end=" ", flush=True)

        seq_results, seq_wall = run_sequential(tasks)
        par_results, par_wall = run_parallel(tasks, n_workers=n_workers)

        seq_agg = aggregate_stats(seq_results, seq_wall, "sequential")
        par_agg = aggregate_stats(par_results, par_wall, "parallel")

        seq_runs.append(seq_agg)
        par_runs.append(par_agg)
        last_seq_results = seq_results
        last_par_results = par_results

        print(
            f"seq={seq_wall:.1f} ms  par={par_wall:.1f} ms  "
            f"speedup={seq_wall/par_wall:.2f}×"
        )

    return {
        "sequential":       seq_runs,
        "parallel":         par_runs,
        "seq_room_results": last_seq_results,
        "par_room_results": last_par_results,
        "room_sequence":    room_sequence,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_pair(
    ax: plt.Axes,
    seq_vals: list[float],
    par_vals: list[float],
    title: str,
    ylabel: str,
    subtitle: str = "",
) -> None:
    """Draw a grouped bar chart comparing sequential vs parallel."""
    labels = ["Sequential", "Parallel"]
    means = [float(np.mean(seq_vals)), float(np.mean(par_vals))]
    errs  = [float(np.std(seq_vals)),  float(np.std(par_vals))]
    colors = [PALETTE["sequential"], PALETTE["parallel"]]

    bars = ax.bar(labels, means, yerr=errs, color=colors, alpha=0.88,
                  capsize=5, width=0.5)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10, pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Annotate bars
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(errs) * 0.05,
            f"{mean:.1f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )


def _speedup_plot(
    ax: plt.Axes,
    seq_walls: list[float],
    par_walls: list[float],
) -> None:
    """Plot speedup (seq / par) per run as a line + scatter."""
    speedups = [s / p for s, p in zip(seq_walls, par_walls)]
    runs = list(range(1, len(speedups) + 1))
    mean_su = float(np.mean(speedups))

    ax.plot(runs, speedups, "o-", color=PALETTE["speedup"], lw=2, ms=6, label="per run")
    ax.axhline(mean_su, color=PALETTE["speedup"], lw=1.2, ls="--",
               label=f"mean {mean_su:.2f}×")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_title("Parallel Speedup\n(sequential / parallel wall time)", fontsize=10)
    ax.set_xlabel("Run", fontsize=9)
    ax.set_ylabel("Speedup ×", fontsize=9)
    ax.set_xticks(runs)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def _per_room_plot(
    ax: plt.Axes,
    seq_results: list[dict],
    par_results: list[dict],
    room_sequence: list[str],
) -> None:
    """Grouped bars showing per-room A* time for seq vs parallel."""
    n = len(room_sequence)
    x = np.arange(n)
    width = 0.35

    seq_times = [r["time_ms"] for r in seq_results]
    par_times = [r["time_ms"] for r in par_results]

    ax.bar(x - width / 2, seq_times, width, label="Sequential",
           color=PALETTE["sequential"], alpha=0.88)
    ax.bar(x + width / 2, par_times, width, label="Parallel",
           color=PALETTE["parallel"], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(room_sequence, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Time (ms)", fontsize=9)
    ax.set_title("Per-Room A* Planning Time\n(last run)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------

def plot_benchmark(
    data: dict,
    output_dir: Path,
    show: bool = False,
    timestamp: str = "",
) -> Path:
    """
    Render benchmark charts and save to *output_dir*.

    Parameters
    ----------
    data:
        Output dict from :func:`run_benchmark`.
    output_dir:
        Directory where PNGs are saved.
    show:
        Whether to call ``plt.show()`` (useful in interactive sessions).
    timestamp:
        String suffix for filenames; auto-generated if empty.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if not timestamp:
        timestamp = time.strftime("%Y%m%d_%H%M%S")

    seq_runs     = data["sequential"]
    par_runs     = data["parallel"]
    seq_room_res = data["seq_room_results"]
    par_room_res = data["par_room_results"]
    room_seq     = data["room_sequence"]

    seq_walls  = [r["wall_time_ms"]   for r in seq_runs]
    par_walls  = [r["wall_time_ms"]   for r in par_runs]
    seq_sums   = [r["sum_time_ms"]    for r in seq_runs]
    par_sums   = [r["sum_time_ms"]    for r in par_runs]
    seq_lens   = [r["total_length_m"] for r in seq_runs]
    par_lens   = [r["total_length_m"] for r in par_runs]
    seq_exp    = [r["total_expanded"] for r in seq_runs]
    par_exp    = [r["total_expanded"] for r in par_runs]

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    ax_wall   = fig.add_subplot(gs[0, 0])
    ax_sum    = fig.add_subplot(gs[0, 1])
    ax_speed  = fig.add_subplot(gs[0, 2])
    ax_room   = fig.add_subplot(gs[1, 0:2])
    ax_len    = fig.add_subplot(gs[1, 2])

    _bar_pair(ax_wall, seq_walls, par_walls,
              "Wall-Clock Time", "ms", "end-to-end latency")
    _bar_pair(ax_sum,  seq_sums,  par_sums,
              "Sum of Worker Times", "ms", "total CPU work")
    _speedup_plot(ax_speed, seq_walls, par_walls)
    _per_room_plot(ax_room, seq_room_res, par_room_res, room_seq)
    _bar_pair(ax_len, seq_lens, par_lens,
              "Total Path Length", "m", "lower = shorter route")

    n_workers = os.cpu_count() or 1
    n_runs    = len(seq_runs)
    fig.suptitle(
        f"Parallel A* Benchmark  ·  {len(room_seq)} rooms  ·  "
        f"{n_workers} CPU cores  ·  {n_runs} runs",
        fontsize=13, fontweight="bold", y=0.98,
    )

    out_path = output_dir / f"benchmark_{timestamp}.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"\n  Chart saved → {out_path.resolve()}")
    return out_path


def print_summary(data: dict) -> None:
    """Print a concise text summary of the benchmark results."""
    seq = data["sequential"]
    par = data["parallel"]
    seq_walls = [r["wall_time_ms"] for r in seq]
    par_walls = [r["wall_time_ms"] for r in par]
    speedups  = [s / p for s, p in zip(seq_walls, par_walls)]

    def _fmt(vals: list[float]) -> str:
        return f"{np.mean(vals):.1f} ± {np.std(vals):.1f} ms"

    print("\n" + "=" * 54)
    print("  Benchmark Summary")
    print("=" * 54)
    print(f"  Rooms traversed : {len(data['room_sequence'])}")
    print(f"  Runs            : {len(seq)}")
    print(f"  Workers         : {os.cpu_count() or 1}")
    print(f"  Sequential wall : {_fmt(seq_walls)}")
    print(f"  Parallel wall   : {_fmt(par_walls)}")
    print(f"  Speedup         : {np.mean(speedups):.2f}× "
          f"(max {max(speedups):.2f}×)")
    print(f"  Path length     : "
          f"{np.mean([r['total_length_m'] for r in par]):.2f} m (parallel)")
    print("=" * 54)
