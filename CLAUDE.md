# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Two layers under one repo:

1. **Gesture / voice drone control** (`playground/`, `readme.md`) — the original
   goal: MediaPipe hand landmarks + (later) STT/LLM → DJI Tello flight commands.
   Lives in `playground/DJITelloPy/` (Tello SDK) and `playground/Tello-LLM/`.
2. **Autonomous 3D navigation** (`3D-segmentation/`, `docs/project-overview.md`) —
   the current active work: natural language → LLM intent parsing → 3D object
   localization in a prebuilt scene → (future) path planning → drone execution.
   This is where almost all recent commits land.

The 3D localization layer has **two complementary object-finding backends**:

- **Mosaic3D path** (per-point, open-vocab): SpUNet101 + CLIP text head produces a
  per-point CLIP-aligned feature. A text query → cosine-sim heatmap → DBSCAN
  spatial clustering → ranked 3D candidates. Open vocabulary (any prompt).
- **UniDet3D path** (per-bbox, closed-set): a 3D object detector emits bounding
  boxes + class labels; each box's class is CLIP-embedded and matched to the query.
  Closed class set per dataset head, but gives clean boxes.

Both consume the same scene point clouds and both output WORLD-frame coordinates
for a downstream path planner.

## Submodules

`unidet3d/` and `Mosaic3D/` are git submodules (see `.gitmodules`) — upstream
research repos, not pip packages. After clone: `git submodule update --init`.
UniDet3D is imported by adding its root to `sys.path` (it self-registers mmdet3d
modules); never expect `pip install unidet3d` to work.

## Two conda environments (they conflict — keep separate)

| env | for | key pins |
|---|---|---|
| `mosaic3d` | Mosaic3D inference + all `webapp*/` viser apps | torch 2.2.2 + cu121, spconv-cu120, open_clip, viser, `transformers<5`, `setuptools<81` |
| `unidet3d` | UniDet3D detection (`miny-det/`, `unidet3d_only/`, webapp_llm UniDet3D mode) | torch 2.1.2 + cu121, mmdet3d 1.4.0, mmcv 2.1.0, MinkowskiEngine, spconv-cu120 2.3.6 |

`mosaic3d` is built by `3D-segmentation/setup_env/setup_env.sh` (pins in
`setup_env/requirements-inference.txt`). The `unidet3d` env build is manual —
exact recipe in `3D-segmentation/webapp_llm/README.md` §1; MinkowskiEngine build
is the fragile step. Activate with:
`source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh && conda activate <env>`.

> **Running BOTH detection backends in one webapp process.** The webapps' Mosaic3D
> path at *serve* time only needs open_clip + viser + sklearn-DBSCAN over the
> precomputed `feat.npy` cache — it never runs spconv/SpUNet101 (that's offline
> `inference/run_inference.py`). So both the Mosaic3D and UniDet3D backends can run
> live in ONE process **if the server is launched from the `unidet3d` env** (torch
> 2.1.2) after `pip install open_clip_torch viser scikit-learn transformers
> accelerate` there. From the `mosaic3d` env the UniDet3D stack won't import, so the
> `Backend` dropdown silently degrades to mosaic3d-only. You cannot install both
> stacks in one env (torch 2.2.2 vs 2.1.2 + mmdet3d/MinkowskiEngine ABI).

> Hardcoded absolute paths appear in two roots: `/data1/workspaces/jgshin22/Gesture-Drone-Control`
> (READMEs, setup) and `/home/jgshin22/work/Gesture-Drone-Control` (scripts in
> `miny-det/`, `unidet3d_detector.py` defaults). Both are checked-out working dirs
> of this repo. When editing scripts, match the path style already in that file.

## Common commands

Long-running steps are launched in **tmux** sessions (`mosaic-env`,
`mosaic-precompute`, `mosaic-web`) — see `3D-segmentation/README.md` §1–6.

```bash
# --- Mosaic3D per-point feature cache (input to heatmap + clustering) ---
# one Matterport house, all regions:
bash 3D-segmentation/scripts/run_precompute.sh <houseID> cuda:0
# single region/scene from ckpt:
python 3D-segmentation/inference/run_inference.py \
  --ckpt data/spunet101.ckpt --data-dir <dir> --out-dir cache/feat_hm3d \
  --regions <region> --device cuda:0

# --- heatmap -> DBSCAN candidate extraction (CLI) ---
python 3D-segmentation/inference/cluster_candidates.py \
  --region <region> --query "a tv" --top-percentile 95 --eps 0.25 \
  --min-points 40 --top-k 5 --device cuda:1     # prints JSON to stdout

# --- viser web apps (mosaic3d env, OR unidet3d env if using the UniDet3D backend) ---
python 3D-segmentation/webapp/server.py --port 8080 --host 0.0.0.0      # base viewer + cluster mode
python 3D-segmentation/webapp_llm/server.py --port 8090 \              # + local-LLM intent parsing
  --llm-model Qwen/Qwen2.5-3B-Instruct --llm-device cuda:1 --clip-device cuda:0
#   Both apps have a `Backend` dropdown (mosaic3d | unidet3d). To enable the
#   UniDet3D option add --enable-unidet3d --unidet3d-dataset scannetpp --unidet3d-device cuda:0
#   and LAUNCH FROM THE unidet3d ENV (see env table note); it runs on the loaded cache scene.

# --- standalone UniDet3D (unidet3d env) ---
python 3D-segmentation/unidet3d_only/server.py        # detection-only viser app
python miny-det/convert.py                            # coord/color/normal .npy -> (N,9) .bin
python miny-det/infer.py                              # .bin -> detections .pkl (edit consts at top)
```

There is no test suite or linter wired up at the repo root; this is research code.

## Architecture notes that span files

- **`webapp_llm/server.py` reuses `webapp/server.py` by import**, not copy — it
  pulls `RegionAssets`, `TextEncoder`, `load_region`, `query_single`,
  `cluster_palette`, etc. from the base webapp. The base `webapp/server.py` is the
  canonical place for scene loading, CLIP text encoding, heatmap coloring, and
  bbox rendering; do not duplicate those — import them. Editing `webapp/server.py`
  signatures can break `webapp_llm`.

- **Scene/feature cache layout** (`3D-segmentation/cache/`): `feat/<region>/`
  holds `feat.npy` (N,768 fp16, CLIP-text-aligned) + `coord.npy` (N,3 float32,
  **raw world coords**). `match/<region>/` holds mesh vertices/colors for
  rendering. The webapp scans this dir **once at startup** — adding a region means
  restarting the server.

- **Coordinate frames matter for the planner.** `coord.npy` and
  `cluster_candidates.py` output are in raw world meters. The webapp subtracts the
  scene bbox center only for camera display; candidate `center` from the cluster
  module is already planner-ready. In `webapp_llm`, the panel's `world=(x,y,z)` is
  the planner value (`asset.center + display_center`). UniDet3D bboxes are in world
  frame too (coords never modified). Read `docs/clustering-candidates.md` before
  touching ranking/coords — it documents why score abs-values are meaningless
  (compare only relative ranks within one query) and the `mean_score·√n_points`
  ranking choice.

- **Dual detection backend in the webapps.** Both `webapp/server.py` and
  `webapp_llm/server.py` expose a `Backend` dropdown (mosaic3d | unidet3d) via the
  shared module `3D-segmentation/webapp/unidet3d_backend.py` (`add_unidet3d_args`,
  `make_detector`, `detect_scene`, `match_boxes`, `render_boxes`). UniDet3D runs on
  the **currently-loaded cache scene**: it feeds `asset.coord` + `asset.vertex_colors`
  (rescaled `/127.5-1` like `miny-det/convert.py`) as the `(N,6)` xyz+rgb input —
  no `.bin`, no normals reload. Detection is cached per scene; boxes render in the
  centered display frame, world coords reported as `box_center + asset.center`.
  `miny-det/` is left untouched as the standalone reference.

- **UniDet3D wrapper** (`3D-segmentation/unidet3d_only/unidet3d_detector.py`) is
  the reusable, lazy-loaded class; `miny-det/infer.py` is the older standalone
  script it was refactored from (consts hardcoded at top). Heavy mmdet3d imports
  are deferred to first `detect()` so the webapp can start without the unidet3d
  env when the mode is off. Input is `(N,9) = xyz,rgb,normal` float32 `.bin`; only
  xyz+rgb are used. The active class head is selected by `dataset_name`
  (scannet / s3dis / multiscan / 3rscan / scannetpp / arkitscenes) — `scannetpp`
  classes are hardcoded as `SCANNETPP_CLASSES`, others resolved from config.

- **LLM intent parsing** (`webapp_llm/llm_parser.py`): a local HF model emits JSON
  `{target_object, clip_prompt, location_hint, action, return_home}`; only
  `clip_prompt` feeds the CLIP/heatmap or UniDet3D match stage. `location_hint`
  (e.g. "옆 방") is **not yet used** — clustering is single-region only (see
  `docs/clustering-candidates.md` §8 limitations).

- **Mosaic3D model specifics** (from `3D-segmentation/README.md` §9): SparseUNet-101,
  out_dim 768 (CLIP ViT-L text dim), PPT condition fixed to `ScanNet`, CLIP text
  encoder `hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP`, inference grid_size 0.02,
  color normalized `/127.5 - 1`.

## Docs to read first for a given task

- Whole-system flow & TODOs: `docs/project-overview.md`
- Candidate extraction / coords / ranking: `docs/clustering-candidates.md`
- Running the Mosaic3D pipeline end-to-end: `3D-segmentation/README.md`
- LLM + UniDet3D webapp + the unidet3d env recipe: `3D-segmentation/webapp_llm/README.md`
- UniDet3D / mmdet3d install background: `docs/mmdet_get_started.md`
