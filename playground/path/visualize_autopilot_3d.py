"""
Render PNG reports for a 3D autopilot flight: obstacles + planned path + flown
trajectory + intrusion points, so collisions can be checked visually offline.

Example:
    python visualize_autopilot_3d.py ^
        --map-json TEEsavR23oF_voxel_map_3d.json ^
        --trajectory-json traj.json ^
        --output-prefix flight_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from unity_scene_voxel import (  # noqa: E402
    UnitySceneVoxel,
    _nearest_occupied_distance,
    load_unity_scene_voxel,
)


def load_points(payload: dict, key: str) -> np.ndarray:
    entries = payload.get(key) or []
    return np.array([[p["x"], p["y"], p["z"]] for p in entries], dtype=float)


def occupied_world_centers(scene_voxel: UnitySceneVoxel) -> np.ndarray:
    zs, ys, xs = np.nonzero(scene_voxel.map_.grid)
    if len(xs) == 0:
        return np.empty((0, 3))
    centers = [scene_voxel.grid_to_world(int(x), int(y), int(z)) for x, y, z in zip(xs, ys, zs)]
    return np.array(centers, dtype=float)


def intrusion_mask(scene_voxel: UnitySceneVoxel, points: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(points), dtype=bool)
    for i, point in enumerate(points):
        grid = scene_voxel.world_to_grid(*point)
        if scene_voxel.map_.is_in_bounds(grid) and not scene_voxel.map_.is_free(grid):
            mask[i] = True
    return mask


def plot_3d(
    output_path: Path,
    obstacles: np.ndarray,
    path: np.ndarray,
    traj: np.ndarray,
    intrusions: np.ndarray,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection="3d")

    if len(obstacles):
        # Subsample very dense maps so the PNG stays readable.
        step = max(1, len(obstacles) // 6000)
        obs = obstacles[::step]
        ax.scatter(obs[:, 0], obs[:, 2], obs[:, 1], s=2, c="#9aa5b1", alpha=0.18, label="obstacles")

    if len(path):
        ax.plot(path[:, 0], path[:, 2], path[:, 1], "-", c="#e53e3e", lw=2.2, label="planned path")
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 2], traj[:, 1], "-", c="#2b6cb0", lw=1.6, label="flown trajectory")
        ax.scatter(*traj[0][[0, 2, 1]], c="#38a169", s=90, marker="^", label="start")
        ax.scatter(*traj[-1][[0, 2, 1]], c="#805ad5", s=90, marker="v", label="end")
    if len(intrusions):
        ax.scatter(
            intrusions[:, 0], intrusions[:, 2], intrusions[:, 1],
            c="#c53030", s=120, marker="x", linewidths=3, label=f"intrusions ({len(intrusions)})",
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_zlabel("Y height (m)")
    ax.set_title("3D Autopilot Flight — planned vs flown")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_topdown(
    output_path: Path,
    scene_voxel: UnitySceneVoxel,
    obstacles: np.ndarray,
    path: np.ndarray,
    traj: np.ndarray,
    intrusions: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    if len(obstacles) and len(traj):
        # Only draw obstacle cells overlapping the altitude band the drone used,
        # so the top-down view shows the walls that actually mattered.
        y_min, y_max = traj[:, 1].min() - 1.0, traj[:, 1].max() + 1.0
        band = obstacles[(obstacles[:, 1] >= y_min) & (obstacles[:, 1] <= y_max)]
        ax.scatter(band[:, 0], band[:, 2], s=14, c="#9aa5b1", alpha=0.5, marker="s",
                   label=f"obstacles (y {y_min:.0f}..{y_max:.0f}m)")

    if len(path):
        ax.plot(path[:, 0], path[:, 2], "-", c="#e53e3e", lw=2.4, label="planned path")
    if len(traj):
        ax.plot(traj[:, 0], traj[:, 2], "-", c="#2b6cb0", lw=1.8, label="flown trajectory")
        ax.plot(traj[0, 0], traj[0, 2], "^", c="#38a169", ms=12, label="start")
        ax.plot(traj[-1, 0], traj[-1, 2], "v", c="#805ad5", ms=12, label="end")
    if len(intrusions):
        ax.plot(intrusions[:, 0], intrusions[:, 2], "x", c="#c53030", ms=13, mew=3,
                label=f"intrusions ({len(intrusions)})")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Top-down view (X-Z)")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_altitude(
    output_path: Path,
    path: np.ndarray,
    traj: np.ndarray,
    traj_intrusion: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    if len(traj):
        steps = np.arange(len(traj))
        ax.plot(steps, traj[:, 1], "-", c="#2b6cb0", lw=1.8, label="flown altitude")
        if traj_intrusion.any():
            ax.plot(steps[traj_intrusion], traj[traj_intrusion, 1], "x", c="#c53030",
                    ms=11, mew=3, label=f"intrusion steps ({int(traj_intrusion.sum())})")
    if len(path):
        # Spread waypoints across the same step axis for a rough overlay.
        wp_steps = np.linspace(0, max(len(traj) - 1, 1), len(path))
        ax.plot(wp_steps, path[:, 1], "--", c="#e53e3e", lw=1.6, label="planned altitude")

    ax.set_xlabel("control step")
    ax.set_ylabel("Y height (m)")
    ax.set_title("Altitude profile")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_clearance(
    output_path: Path,
    scene_voxel: UnitySceneVoxel,
    traj: np.ndarray,
    traj_intrusion: np.ndarray,
) -> None:
    """Distance from the drone to the nearest occupied voxel center, per step.

    This is the most direct collision evidence: a dip to ~0 means the drone was
    inside an occupied voxel, regardless of what any 3D view looks like.
    """
    if not len(traj):
        return
    clearances = np.array([
        _nearest_occupied_distance(scene_voxel, p, search_radius=5) for p in traj
    ])
    finite = np.isfinite(clearances)
    steps = np.arange(len(traj))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps[finite], clearances[finite], "-", c="#2b6cb0", lw=1.8, label="clearance to nearest obstacle")
    half_voxel = scene_voxel.voxel_size / 2.0
    ax.axhline(half_voxel, color="#dd6b20", ls="--", lw=1.4,
               label=f"voxel boundary (~{half_voxel:.1f}m = touching)")
    ax.axhline(0.0, color="#c53030", ls="-", lw=1.2, label="voxel center (inside obstacle)")
    if traj_intrusion.any():
        ax.plot(steps[traj_intrusion], clearances[traj_intrusion], "x", c="#c53030", ms=11, mew=3,
                label=f"intrusion steps ({int(traj_intrusion.sum())})")

    ax.set_xlabel("control step")
    ax.set_ylabel("distance (m)")
    ax.set_title("Obstacle clearance per step — dips below the orange line mean wall contact")
    ax.set_ylim(bottom=-0.2)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=170)
    plt.close(fig)


def plot_slices(
    output_path: Path,
    scene_voxel: UnitySceneVoxel,
    path: np.ndarray,
    traj: np.ndarray,
    intrusions: np.ndarray,
) -> None:
    """One top-down panel per voxel layer the drone flew through, so each floor's
    walls can be compared against the exact trajectory segment at that height."""
    if not len(traj):
        return
    grid = scene_voxel.map_.grid  # [z, y, x]
    voxel = scene_voxel.voxel_size

    def layer_of(y_world: float) -> int:
        return int(np.floor((y_world - scene_voxel.origin_y) / voxel))

    layers = sorted({layer_of(y) for y in traj[:, 1] if 0 <= layer_of(y) < scene_voxel.map_.height})
    if not layers:
        return

    cols = min(4, len(layers))
    rows_n = int(np.ceil(len(layers) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.4 * cols, 4.6 * rows_n), squeeze=False)

    for idx, layer in enumerate(layers):
        ax = axes[idx // cols][idx % cols]
        y_lo = scene_voxel.origin_y + layer * voxel
        y_hi = y_lo + voxel

        zs, xs = np.nonzero(grid[:, layer, :])
        if len(xs):
            ox = scene_voxel.origin_x + (xs + 0.5) * voxel
            oz = scene_voxel.origin_z + (zs + 0.5) * voxel
            ax.scatter(ox, oz, s=26, c="#4a5568", alpha=0.75, marker="s")

        in_layer = (traj[:, 1] >= y_lo) & (traj[:, 1] < y_hi)
        if len(path):
            ax.plot(path[:, 0], path[:, 2], "-", c="#e53e3e", lw=1.0, alpha=0.4)
        if in_layer.any():
            ax.plot(traj[in_layer, 0], traj[in_layer, 2], ".-", c="#2b6cb0", lw=2.0, ms=4)
        if len(intrusions):
            hit = (intrusions[:, 1] >= y_lo) & (intrusions[:, 1] < y_hi)
            if hit.any():
                ax.plot(intrusions[hit, 0], intrusions[hit, 2], "x", c="#c53030", ms=13, mew=3)

        ax.set_title(f"layer y {y_lo:.0f}..{y_hi:.0f}m ({int(in_layer.sum())} steps)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    for idx in range(len(layers), rows_n * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle("Per-altitude slices — walls (gray) vs trajectory (blue) at each flight layer", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_interactive_html(
    output_path: Path,
    scene_voxel: UnitySceneVoxel,
    obstacles: np.ndarray,
    path: np.ndarray,
    traj: np.ndarray,
    intrusions: np.ndarray,
) -> None:
    """Self-contained HTML viewer: drag to orbit, wheel to zoom, height slider to
    cut away ceilings. No external libraries, works offline in any browser."""
    max_obstacles = 12000
    step = max(1, len(obstacles) // max_obstacles)
    obs = obstacles[::step]

    data = {
        "voxel": scene_voxel.voxel_size,
        "obstacles": np.round(obs, 2).tolist(),
        "path": np.round(path, 2).tolist() if len(path) else [],
        "traj": np.round(traj, 2).tolist() if len(traj) else [],
        "intrusions": np.round(intrusions, 2).tolist() if len(intrusions) else [],
    }

    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>3D Flight Viewer</title>
<style>
  html,body{margin:0;height:100%;background:#0f1420;color:#dfe6ef;font:13px/1.5 system-ui,sans-serif;overflow:hidden}
  #hud{position:fixed;top:10px;left:12px;background:rgba(15,20,32,.85);padding:10px 14px;border-radius:8px;max-width:330px}
  #hud b{color:#fff} .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
  label{display:block;margin-top:6px}
  input[type=range]{width:180px;vertical-align:middle}
  canvas{display:block}
</style></head><body>
<canvas id="c"></canvas>
<div id="hud">
  <b>3D Flight Viewer</b><br>
  drag: rotate &nbsp; wheel: zoom &nbsp; shift+drag: pan<br>
  <span class="sw" style="background:#9aa5b1"></span>obstacles
  <span class="sw" style="background:#ff5252"></span>planned
  <span class="sw" style="background:#4fc3f7"></span>flown
  <span class="sw" style="background:#ff1744"></span>intrusion<br>
  <label>height cut: <input id="ymax" type="range" min="0" max="100" value="100">
  <span id="ymaxv"></span></label>
</div>
<script>
const D = __DATA__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
let W, H; function resize(){ W=cv.width=innerWidth; H=cv.height=innerHeight; } resize();
addEventListener('resize', ()=>{resize(); draw();});

const all = D.obstacles.concat(D.path, D.traj);
const mins=[1e9,1e9,1e9], maxs=[-1e9,-1e9,-1e9];
for (const p of all) for (let i=0;i<3;i++){ mins[i]=Math.min(mins[i],p[i]); maxs[i]=Math.max(maxs[i],p[i]); }
const center=[(mins[0]+maxs[0])/2,(mins[1]+maxs[1])/2,(mins[2]+maxs[2])/2];
const span=Math.max(maxs[0]-mins[0],maxs[1]-mins[1],maxs[2]-mins[2]);

let yaw=0.8, pitch=0.5, zoom=1.0, panX=0, panY=0, yCut=maxs[1];
function project(p){
  const x=p[0]-center[0], y=p[1]-center[1], z=p[2]-center[2];
  const cy=Math.cos(yaw), sy=Math.sin(yaw), cp=Math.cos(pitch), sp=Math.sin(pitch);
  const rx=cy*x+sy*z, rz=-sy*x+cy*z;
  const ry=cp*y-sp*rz;
  const s=(Math.min(W,H)*0.85/span)*zoom;
  return [W/2+rx*s+panX, H/2-ry*s+panY];
}
function polyline(points, color, width){
  if(points.length<2) return;
  ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath();
  let started=false;
  for(const p of points){
    const q=project(p);
    if(!started){ ctx.moveTo(q[0],q[1]); started=true; } else ctx.lineTo(q[0],q[1]);
  }
  ctx.stroke();
}
function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle='rgba(154,165,177,0.35)';
  for(const p of D.obstacles){
    if(p[1]>yCut) continue;
    const q=project(p);
    ctx.fillRect(q[0]-1.2,q[1]-1.2,2.4,2.4);
  }
  polyline(D.path,'#ff5252',2.5);
  polyline(D.traj,'#4fc3f7',1.8);
  ctx.fillStyle='#ff1744';
  for(const p of D.intrusions){
    const q=project(p);
    ctx.beginPath(); ctx.arc(q[0],q[1],6,0,7); ctx.fill();
    ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.stroke();
  }
  if(D.traj.length){
    const s=project(D.traj[0]), e=project(D.traj[D.traj.length-1]);
    ctx.fillStyle='#69f0ae'; ctx.beginPath(); ctx.arc(s[0],s[1],7,0,7); ctx.fill();
    ctx.fillStyle='#b388ff'; ctx.beginPath(); ctx.arc(e[0],e[1],7,0,7); ctx.fill();
  }
}
let dragging=false, panning=false, lx=0, ly=0;
cv.addEventListener('mousedown',e=>{dragging=true; panning=e.shiftKey; lx=e.clientX; ly=e.clientY;});
addEventListener('mouseup',()=>dragging=false);
addEventListener('mousemove',e=>{
  if(!dragging) return;
  const dx=e.clientX-lx, dy=e.clientY-ly; lx=e.clientX; ly=e.clientY;
  if(panning){ panX+=dx; panY+=dy; }
  else { yaw+=dx*0.008; pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*0.008)); }
  draw();
});
cv.addEventListener('wheel',e=>{ e.preventDefault(); zoom*=e.deltaY<0?1.1:0.9; draw(); },{passive:false});
const slider=document.getElementById('ymax'), label=document.getElementById('ymaxv');
slider.addEventListener('input',()=>{
  yCut=mins[1]+(maxs[1]-mins[1])*slider.value/100;
  label.textContent=' y ≤ '+yCut.toFixed(1)+'m';
  draw();
});
label.textContent=' y ≤ '+maxs[1].toFixed(1)+'m';
draw();
</script></body></html>
"""
    html = html.replace("__DATA__", json.dumps(data))
    output_path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-json", required=True, help="Exported 3D voxel map JSON.")
    parser.add_argument("--trajectory-json", required=True,
                        help="Trajectory JSON produced by unity_autopilot_3d.py --trajectory-out.")
    parser.add_argument("--output-prefix", default="flight_report",
                        help="Prefix for the generated PNG files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_voxel = load_unity_scene_voxel(args.map_json)
    payload = json.loads(Path(args.trajectory_json).read_text(encoding="utf-8"))

    traj = load_points(payload, "trajectory_world")
    path = load_points(payload, "path_world")
    obstacles = occupied_world_centers(scene_voxel)

    traj_mask = intrusion_mask(scene_voxel, traj) if len(traj) else np.zeros(0, dtype=bool)
    intrusions = traj[traj_mask] if len(traj) else np.empty((0, 3))

    prefix = Path(args.output_prefix)
    plot_3d(prefix.with_name(prefix.name + "_3d.png"), obstacles, path, traj, intrusions)
    plot_topdown(prefix.with_name(prefix.name + "_topdown.png"), scene_voxel, obstacles, path, traj, intrusions)
    plot_altitude(prefix.with_name(prefix.name + "_altitude.png"), path, traj, traj_mask)
    plot_clearance(prefix.with_name(prefix.name + "_clearance.png"), scene_voxel, traj, traj_mask)
    plot_slices(prefix.with_name(prefix.name + "_slices.png"), scene_voxel, path, traj, intrusions)
    generate_interactive_html(prefix.with_name(prefix.name + "_viewer.html"), scene_voxel, obstacles, path, traj, intrusions)

    print(f"trajectory points: {len(traj)}, intrusion steps: {int(traj_mask.sum())}")
    for suffix in ("_3d.png", "_topdown.png", "_altitude.png", "_clearance.png", "_slices.png", "_viewer.html"):
        print(f"saved {prefix.with_name(prefix.name + suffix)}")


if __name__ == "__main__":
    main()
