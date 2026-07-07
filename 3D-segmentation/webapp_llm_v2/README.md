# webapp_llm_v2 — NL query → localize → path plan → Tello SDK export

Closes the loop from a natural-language drone command to a **DJI Tello SDK
command program**, for houses **00800** and **00809** only.

```
terminal query
  → local LLM intent parse          (webapp_llm.llm_parser)
  → CLIP heatmap + DBSCAN candidate  (webapp.server + inference.cluster_candidates)
  → viser: heatmap + target marker
  → A* / RRT* path, current pos → target   (webapp_llm_v2/planner.py)
  → viser: path polyline + start marker
  → Tello command program written to out/   (webapp_llm_v2/sdk_export.py)
```

Unlike `webapp_llm/`, the query is typed in the **terminal**, not the viser GUI;
viser is used for visualization only. It runs the **mosaic3d backend only**
(UniDet3D excluded).

## Run

Runs in the **`mosaic3d`** conda env — no unidet3d / mmdet3d needed. Requires
`viser`, `open_clip_torch`, `transformers`, `scipy` (all present in `mosaic3d`).

```bash
source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh
conda activate mosaic3d
python 3D-segmentation/webapp_llm_v2/server.py \
    --building 00809_Qpor2mEya8F \
    --clip-device cuda:0 --llm-device cuda:1
```

Open the printed viser URL, then type queries in the terminal, e.g.
`주방 냉장고 찾아줘`, `002_011 방에 있는 tv 사진 찍어와줘`.

REPL commands: `home` (reset start to launch point), `building <id>` (switch
between `00800_TEEsavR23oF` / `00809_Qpor2mEya8F`), `quit`.

### viser markers
- **gold** sphere = `HOME` (launch / return point; `--home-xyz` or default).
- **cyan** sphere = `DRONE` — current position; starts at home, moves to each
  query's goal (continuous mission), resets on `home`.
- **red** = localized target + bbox, **green** = planned path, **blue** = path
  start.
- **Show room labels** checkbox (viser GUI) toggles the per-room ID labels.

### Key flags
- `--algo astar|rrt` — A* (default) or RRT*.
- `--resolution 0.15 --margin 1 --sample 10` — voxel grid (obstacle inflation,
  point subsample).
- `--home-xyz X Y Z` — world-meter launch point (default: lowest-floor first
  room centroid).
- `--tello-speed 40` — cm/s in emitted `go_xyz_speed` commands.
- `--out-dir` — where the JSON programs are written (default `out/`).

## Continuous mission

The **first** query plans from `home`; **each subsequent** query plans from the
**previous query's goal**. `home` resets it; switching buildings resets it.

## Output format

Each query writes `out/<timestamp>_<target>.json`, mirroring the `gyucheol`
branch's Tello contract — Ollama-style tool-call dicts
`{"function": {"name", "arguments"}, "sdk": "<raw tello cmd>"}` that a downstream
layer can dispatch with `getattr(drone_tools, name)(**arguments)`:

```json
{
  "meta": { "query": "...", "target_object": "refrigerator", "action": "goto",
            "return_home": false, "algo": "astar", "building": "00809_Qpor2mEya8F",
            "home_world": [...], "start_world": [...], "goal_world": [...],
            "path_length_m": 7.4, "n_waypoints": 9, "speed": 40,
            "world_to_body": "fixed-heading ... meters*100 -> cm" },
  "waypoints_world_m": [[x,y,z], ...],
  "commands": [
    {"function": {"name": "takeoff",      "arguments": {}}, "sdk": "takeoff"},
    {"function": {"name": "go_xyz_speed", "arguments": {"x": 120, "y": -30, "z": 0, "speed": 40}},
     "sdk": "go 120 -30 0 40"},
    {"function": {"name": "land",         "arguments": {}}, "sdk": "land"}
  ]
}
```

`go_xyz_speed` (real Tello SDK, `tello.py`) is used for 3D path segments: relative
**cm** per axis in `[-500, 500]`, speed cm/s in `[10, 100]`. Near-zero segments
(all axes < 20 cm, which the SDK rejects) are merged; axes over 500 cm are split.
`action == "take_photo"` appends `streamon` + a `take_photo` marker (gyucheol has
no still-capture tool — real capture is `cv2.imwrite(get_frame_read().frame)`).
`return_home` appends a reversed leg back to `home_world`.

## Limitations (v1)

- **Single whole-building voxel grid**; every point is an obstacle (no semantic
  free-space carving). Cross-floor A*/RRT* only succeed if the point cloud has a
  real vertical opening (stairwell); otherwise a multi-floor plan can fail.
- **World→body heading is fixed** (drone faces +world-x). Real yaw tracking, and
  door-graph routing (`cache/00800/door_graph_dfs.json`; chaewon `rooms_graph.json`
  for 809), are future work — not used here.
- The planner core in `planner.py` is vendored from the `chaewon` branch
  (`comparison/3D.py`), trimmed to the pure-numpy planning functions.
