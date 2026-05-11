"""Viser-based open-vocabulary visualization for Matterport3D regions.

UI flow:
    1. Choose a region (drop-down).
    2. Enter a query in either:
        - "single query" mode  -> heatmap of cosine sim with one prompt.
        - "class list"   mode  -> argmax semantic seg over comma-separated labels.
    3. Mesh is colored accordingly. Original Matterport3D mesh is shown if
       matching cache exists; otherwise raw compressed point cloud is shown.

Cache layout expected (created by inference/run_inference.py + matching/align_to_mesh.py):
    cache/feat/<region>/feat.npy           (N, D) float16
    cache/feat/<region>/coord.npy          (N, 3) float32
    cache/match/<region>/vertex2point.npy  (V,)   int32   (optional)
    cache/match/<region>/vertices.npy      (V, 3) float32 (optional)
    cache/match/<region>/faces.npy         (F, 3) int32   (optional)
    cache/match/<region>/vertex_colors.npy (V, 3) uint8   (optional)
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Re-export for legacy callers in case anything imports Tuple from here.
__all__ = ["main"]

import matplotlib
import numpy as np
import torch
import viser
from open_clip import create_model_and_transforms, get_tokenizer

# Make the sibling `inference/` package importable when launching as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.cluster_candidates import (  # noqa: E402
    ClusterParams,
    candidates_from_heatmap,
)

CLIP_MODEL_ID = "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP"
DEFAULT_PROMPTS = [
    "a chair",
    "a table",
    "a sofa",
    "a bed",
    "a wall",
    "a floor",
]


@dataclass
class RegionAssets:
    name: str
    feat: torch.Tensor  # (N, D), normalized (unit length), index-aligned with coord
    coord: np.ndarray  # (N, 3) point cloud, already centered to origin
    vertex_colors: Optional[np.ndarray] = None  # (N, 3) uint8 — per-point RGB
    center: np.ndarray = None  # (3,) world center subtracted from coord
    # When this asset represents a whole building, region_slices maps each
    # region name to (start, end) indices into coord/feat. None for single-region.
    region_slices: Optional[Dict[str, Tuple[int, int]]] = None

    @property
    def n_render_units(self) -> int:
        return len(self.coord)

    def feat_for_render(self) -> torch.Tensor:
        return self.feat


def list_regions(feat_dir: Path) -> List[str]:
    return sorted(p.name for p in feat_dir.iterdir() if (p / "feat.npy").exists())


def _building_id(region_name: str) -> str:
    """Extract building ID from region name (first two underscore-separated parts)."""
    parts = region_name.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else region_name


def list_buildings(feat_dir: Path) -> List[str]:
    regions = list_regions(feat_dir)
    seen: dict = {}
    for r in regions:
        bid = _building_id(r)
        seen.setdefault(bid, None)
    return sorted(seen.keys())


def regions_for_building(building_id: str, feat_dir: Path) -> List[str]:
    return [r for r in list_regions(feat_dir) if _building_id(r) == building_id]


def _find_point_colors(name: str, n_points: int, feat_dir: Path, match_dir: Path) -> Optional[np.ndarray]:
    """Locate per-point RGB color matching the feature/coord at `feat_dir/<name>`.

    Priority:
        1. cache/match/<name>/vertex_colors.npy   (HM3D: same indexing as feat)
        2. data/matterport3d_compressed/<name>/color.npy   (MP3D compressed)
    Returns (N, 3) uint8 or None.
    """
    rmatch = match_dir / name
    if (rmatch / "vertex_colors.npy").exists():
        vc = np.load(rmatch / "vertex_colors.npy")
        if len(vc) == n_points:
            return vc.astype(np.uint8)[:, :3]
    # Walk up from feat_dir to find the repo root (contains a 'data/' directory).
    repo_root = feat_dir
    for _ in range(8):
        if (repo_root / "data").is_dir():
            break
        repo_root = repo_root.parent
    data_dir = repo_root / "data"
    for split in ("train", "val"):
        color_path = data_dir / "hm3d_compressed" / split / name / "color.npy"
        if color_path.exists():
            c = np.load(color_path)
            if len(c) == n_points:
                return c.astype(np.uint8)[:, :3]
    mp3d_compressed_color = data_dir / "matterport3d_compressed" / name / "color.npy"
    if mp3d_compressed_color.exists():
        c = np.load(mp3d_compressed_color)
        if len(c) == n_points:
            return c.astype(np.uint8)[:, :3]

    return None


def load_region(name: str, feat_dir: Path, match_dir: Path, device: torch.device) -> RegionAssets:
    rfeat = feat_dir / name
    feat = np.load(rfeat / "feat.npy")  # fp16
    feat_t = torch.from_numpy(feat).to(device, dtype=torch.float32)
    feat_t = torch.nn.functional.normalize(feat_t, dim=-1)
    coord = np.load(rfeat / "coord.npy").astype(np.float32, copy=True)

    rgb = _find_point_colors(name, len(coord), feat_dir, match_dir)

    # Center on bbox midpoint so the camera default frames the scene.
    center = ((coord.min(0) + coord.max(0)) / 2).astype(np.float32)
    coord -= center

    return RegionAssets(
        name=name,
        feat=feat_t,
        coord=coord,
        vertex_colors=rgb,
        center=center,
    )


def load_building(building_id: str, feat_dir: Path, match_dir: Path, device: torch.device) -> RegionAssets:
    """Load all regions of a building and merge into one RegionAssets."""
    region_names = regions_for_building(building_id, feat_dir)
    if not region_names:
        raise ValueError(f"No regions found for building {building_id}")

    all_feat: List[torch.Tensor] = []
    all_coord: List[np.ndarray] = []
    all_rgb: List[Optional[np.ndarray]] = []
    region_slices: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    for name in region_names:
        rfeat = feat_dir / name
        feat = np.load(rfeat / "feat.npy")  # float16 on disk
        # Normalize on CPU to avoid OOM; keep as float16 to save RAM.
        feat_t = torch.from_numpy(feat).float()
        feat_t = torch.nn.functional.normalize(feat_t, dim=-1).half()
        coord = np.load(rfeat / "coord.npy").astype(np.float32, copy=True)
        rgb = _find_point_colors(name, len(coord), feat_dir, match_dir)
        all_feat.append(feat_t)
        all_coord.append(coord)
        all_rgb.append(rgb)
        region_slices[name] = (cursor, cursor + len(coord))
        cursor += len(coord)

    merged_coord = np.concatenate(all_coord, axis=0)
    merged_feat = torch.cat(all_feat, dim=0)  # CPU float16

    has_rgb = all(r is not None for r in all_rgb)
    if has_rgb:
        merged_rgb: Optional[np.ndarray] = np.concatenate(all_rgb, axis=0)  # type: ignore[arg-type]
    else:
        merged_rgb = None

    center = ((merged_coord.min(0) + merged_coord.max(0)) / 2).astype(np.float32)
    merged_coord -= center

    print(f"[building] {building_id}: {len(region_names)} regions, {len(merged_coord)} points")
    return RegionAssets(
        name=building_id,
        feat=merged_feat,
        coord=merged_coord,
        vertex_colors=merged_rgb,
        center=center,
        region_slices=region_slices,
    )


class TextEncoder:
    """Cached CLIP text encoder (open_clip)."""

    def __init__(self, model_id: str, device: torch.device):
        self.device = device
        print(f"[clip] loading {model_id} ...")
        self.model, _, _ = create_model_and_transforms(model_id, device=device)
        self.tokenizer = get_tokenizer(model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self._cache: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, prompts: List[str]) -> torch.Tensor:
        miss = [p for p in prompts if p not in self._cache]
        if miss:
            tok = self.tokenizer(miss).to(self.device)
            feats = self.model.encode_text(tok)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            for p, f in zip(miss, feats):
                self._cache[p] = f
        return torch.stack([self._cache[p] for p in prompts], dim=0)


def heatmap_colors(scores: np.ndarray, low_pct: float = 5, high_pct: float = 99) -> np.ndarray:
    """Map (N,) scores → (N,3) uint8 RGB using turbo colormap with percentile stretch."""
    if scores.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    lo, hi = np.percentile(scores, [low_pct, high_pct])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((scores - lo) / (hi - lo), 0, 1)
    cmap = matplotlib.colormaps.get_cmap("turbo")
    rgb = (cmap(norm)[:, :3] * 255).astype(np.uint8)
    return rgb


def class_colors(num_classes: int) -> np.ndarray:
    cmap = matplotlib.colormaps.get_cmap("tab20")
    return (cmap(np.linspace(0, 1, max(num_classes, 1)))[:, :3] * 255).astype(np.uint8)


def cluster_palette(k: int) -> np.ndarray:
    """Distinct, saturated colors for cluster rank markers."""
    cmap = matplotlib.colormaps.get_cmap("Set1")
    return (cmap(np.linspace(0, 1, max(k, 1)))[:, :3] * 255).astype(np.uint8)


def _bbox_edge_segments(bb_min: np.ndarray, bb_max: np.ndarray) -> np.ndarray:
    """Return (12, 2, 3) line segments tracing the 12 edges of an axis-aligned bbox."""
    x0, y0, z0 = bb_min
    x1, y1, z1 = bb_max
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]
    return np.stack([np.stack([corners[a], corners[b]]) for a, b in edges], axis=0)


_CHUNK = 200_000  # points per GPU chunk to avoid OOM on large buildings


def _sim_chunked(feat_cpu: torch.Tensor, text_feat_gpu: torch.Tensor) -> torch.Tensor:
    """Compute cosine sim in chunks; feat_cpu may be CPU float16."""
    device = text_feat_gpu.device
    results = []
    for i in range(0, len(feat_cpu), _CHUNK):
        chunk = feat_cpu[i : i + _CHUNK].to(device, dtype=torch.float32)
        results.append((chunk @ text_feat_gpu.unsqueeze(-1)).squeeze(-1).cpu())
    return torch.cat(results, dim=0)


def _sim_classes_chunked(feat_cpu: torch.Tensor, text_feats_gpu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = text_feats_gpu.device
    all_idx, all_score = [], []
    for i in range(0, len(feat_cpu), _CHUNK):
        chunk = feat_cpu[i : i + _CHUNK].to(device, dtype=torch.float32)
        sim = chunk @ text_feats_gpu.t()
        score, idx = sim.max(dim=1)
        all_idx.append(idx.cpu())
        all_score.append(score.cpu())
    return torch.cat(all_idx), torch.cat(all_score)


def query_single(
    asset: RegionAssets, text_feat: torch.Tensor
) -> np.ndarray:
    """Cosine sim with one prompt → per-render-unit score (NumPy)."""
    feat = asset.feat_for_render()
    if feat.device == text_feat.device:
        sim = (feat @ text_feat.unsqueeze(-1)).squeeze(-1)
        return sim.detach().cpu().numpy()
    return _sim_chunked(feat, text_feat).numpy()


def query_classes(
    asset: RegionAssets, text_feats: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Argmax over class prompts → (labels, top scores)."""
    feat = asset.feat_for_render()
    if feat.device == text_feats.device:
        sim = feat @ text_feats.t()  # (N, K)
        top_score, top_idx = sim.max(dim=1)
        return top_idx.detach().cpu().numpy(), top_score.detach().cpu().numpy()
    top_idx, top_score = _sim_classes_chunked(feat, text_feats)
    return top_idx.numpy(), top_score.numpy()


def upload_points(
    server: viser.ViserServer,
    name: str,
    asset: RegionAssets,
    rgb_uint8: np.ndarray,
    point_size: float,
    *,
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    point_shape: str = "rounded",
) -> None:
    arr = np.asarray(rgb_uint8)
    if arr.ndim != 2 or arr.shape[1] not in (3, 4):
        raise ValueError(f"colors shape must be (N,3|4); got {arr.shape}")
    arr = arr[:, :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    colors_f = (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)
    pts = asset.coord.astype(np.float32)
    if any(offset):
        pts = pts + np.asarray(offset, dtype=np.float32)
    server.scene.add_point_cloud(
        name=name,
        points=pts,
        colors=colors_f,
        point_size=point_size,
        point_shape=point_shape,
    )


def blend_with_rgb(
    overlay_uint8: np.ndarray, base_uint8: np.ndarray, alpha: float
) -> np.ndarray:
    """alpha=1 -> pure overlay; alpha=0 -> pure base (RGB)."""
    a = float(np.clip(alpha, 0.0, 1.0))
    o = overlay_uint8.astype(np.float32)
    b = base_uint8.astype(np.float32)
    out = a * o + (1.0 - a) * b
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feat-dir",
        default=str(
            Path(__file__).resolve().parents[1] / "cache" / "feat"
        ),
    )
    ap.add_argument(
        "--match-dir",
        default=str(
            Path(__file__).resolve().parents[1] / "cache" / "match"
        ),
    )
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    feat_dir = Path(args.feat_dir)
    match_dir = Path(args.match_dir)
    if not feat_dir.exists():
        raise SystemExit(f"No feat dir: {feat_dir} (run inference first)")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    regions = list_regions(feat_dir)
    if not regions:
        raise SystemExit(f"No regions found under {feat_dir}")

    buildings = list_buildings(feat_dir)

    print(f"[viser] available regions: {regions}")
    print(f"[viser] available buildings: {buildings}")
    text_encoder = TextEncoder(CLIP_MODEL_ID, device)

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.world_axes.visible = True

    state: Dict[str, object] = {
        "asset": None,
        "cluster_handles": [],
        "region_label_handles": [],
    }

    view_level = server.gui.add_dropdown(
        "View level", options=["building", "room"], initial_value="building"
    )
    building_dropdown = server.gui.add_dropdown(
        "Building", options=buildings, initial_value=buildings[0]
    )
    region_dropdown = server.gui.add_dropdown(
        "Room", options=regions, initial_value=regions[0]
    )
    region_dropdown.visible = False
    # In building view, select a room to highlight it within the whole house.
    highlight_dropdown = server.gui.add_dropdown(
        "Highlight room (in building)",
        options=["(none)"] + regions,
        initial_value="(none)",
    )
    show_region_labels = server.gui.add_checkbox(
        "Show region IDs (building view)", initial_value=True
    )
    mode = server.gui.add_dropdown(
        "Mode",
        options=["single query", "cluster", "class list", "rgb"],
        initial_value="rgb",
    )
    query_text = server.gui.add_text("Query / classes (comma-separated)", initial_value="a chair")
    threshold = server.gui.add_slider(
        "Threshold (single mode)", min=-0.05, max=0.5, step=0.005, initial_value=0.0
    )
    point_size_slider = server.gui.add_slider(
        "Point size", min=0.005, max=0.10, step=0.005, initial_value=0.02
    )
    overlay_alpha = server.gui.add_slider(
        "Query opacity", min=0.0, max=1.0, step=0.05, initial_value=0.7
    )
    cluster_top_pct = server.gui.add_slider(
        "Cluster: top-percentile", min=80.0, max=99.5, step=0.5, initial_value=95.0
    )
    cluster_eps = server.gui.add_slider(
        "Cluster: eps (m)", min=0.05, max=1.0, step=0.05, initial_value=0.25
    )
    cluster_top_k = server.gui.add_slider(
        "Cluster: top-K", min=1, max=10, step=1, initial_value=5
    )
    cluster_marker_size = server.gui.add_slider(
        "Cluster: marker radius (m)", min=0.05, max=0.5, step=0.01, initial_value=0.15
    )
    apply_btn = server.gui.add_button("Apply")
    status_md = server.gui.add_markdown("_status: ready_")

    legend_md = server.gui.add_markdown("")

    def clear_region_labels() -> None:
        for h in state["region_label_handles"]:  # type: ignore[assignment]
            try:
                h.remove()
            except Exception:
                pass
        state["region_label_handles"] = []

    def render_region_labels(asset: RegionAssets) -> None:
        clear_region_labels()
        if not show_region_labels.value or asset.region_slices is None:
            return
        handles = []
        for name, (s, e) in asset.region_slices.items():
            pts = asset.coord[s:e]
            if len(pts) == 0:
                continue
            centroid = pts.mean(axis=0)
            top_z = float(pts[:, 2].max()) + 0.3
            handles.append(server.scene.add_label(
                f"/region_labels/{name}",
                text=name,
                position=(float(centroid[0]), float(centroid[1]), top_z),
            ))
        state["region_label_handles"] = handles

    def apply_highlight(rgb: np.ndarray, asset: RegionAssets) -> np.ndarray:
        """If a room is selected in the highlight dropdown, recolor that slice."""
        if asset.region_slices is None:
            return rgb
        sel = highlight_dropdown.value
        if sel == "(none)" or sel not in asset.region_slices:
            return rgb
        out = rgb.copy()
        s, e = asset.region_slices[sel]
        # Dim other points, paint selected region a saturated color.
        dimmed = (out.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)
        out = dimmed
        out[s:e] = np.array([255, 64, 64], dtype=np.uint8)  # red highlight
        return out

    def clear_cluster_markers() -> None:
        for h in state["cluster_handles"]:  # type: ignore[assignment]
            try:
                h.remove()
            except Exception:
                pass
        state["cluster_handles"] = []

    def rgb_for_asset(asset: RegionAssets) -> np.ndarray:
        if asset.vertex_colors is not None:
            return asset.vertex_colors
        return np.full((asset.n_render_units, 3), 200, np.uint8)

    def reset_camera_to_asset(asset: RegionAssets) -> None:
        ext = asset.coord.max(0) - asset.coord.min(0)
        diag = float(np.linalg.norm(ext)) or 5.0
        for client in server.get_clients().values():
            client.camera.position = (diag * 0.9, -diag * 0.9, diag * 0.6)
            client.camera.look_at = (0.0, 0.0, 0.0)

    def current_point_size() -> float:
        return float(point_size_slider.value)

    def render_panel(asset: RegionAssets, rgb: np.ndarray) -> None:
        rgb_out = apply_highlight(rgb, asset)
        upload_points(server, "/region", asset, rgb_out, current_point_size())

    def load_and_render(region_name: str) -> None:
        t0 = time.time()
        asset = load_region(region_name, feat_dir, match_dir, device)
        state["asset"] = asset
        clear_cluster_markers()
        clear_region_labels()
        render_panel(asset, rgb_for_asset(asset))
        reset_camera_to_asset(asset)
        status_md.content = (
            f"_loaded **{region_name}**: "
            f"{asset.n_render_units} points, "
            f"world-center=({asset.center[0]:.1f}, {asset.center[1]:.1f}, {asset.center[2]:.1f}) "
            f"({time.time()-t0:.2f}s)_"
        )

    def load_and_render_building(building_id: str) -> None:
        t0 = time.time()
        status_md.content = f"_loading building **{building_id}** ..._"
        asset = load_building(building_id, feat_dir, match_dir, device)
        state["asset"] = asset
        clear_cluster_markers()
        # Restrict highlight dropdown choices to rooms in this building.
        building_rooms = regions_for_building(building_id, feat_dir)
        highlight_dropdown.options = ["(none)"] + building_rooms
        highlight_dropdown.value = "(none)"
        render_panel(asset, rgb_for_asset(asset))
        render_region_labels(asset)
        reset_camera_to_asset(asset)
        n_rooms = len(building_rooms)
        status_md.content = (
            f"_loaded **{building_id}**: {n_rooms} rooms, "
            f"{asset.n_render_units} points "
            f"({time.time()-t0:.2f}s)_"
        )

    def apply_query() -> None:
        asset: RegionAssets = state["asset"]
        if asset is None:
            return
        m = mode.value
        text = query_text.value.strip()

        rgb_base = rgb_for_asset(asset)

        # Markers only persist while we're in cluster mode.
        if m != "cluster":
            clear_cluster_markers()

        if m == "rgb":
            render_panel(asset, rgb_base)
            legend_md.content = ""
            status_md.content = "_rendered RGB_"
            return

        if not text:
            render_panel(asset, rgb_base)
            status_md.content = "_query is empty_"
            return

        if m == "single query":
            tf = text_encoder.encode([text])[0]
            scores = query_single(asset, tf)
            mask = scores >= threshold.value
            overlay = rgb_base.copy()
            if mask.any():
                overlay[mask] = heatmap_colors(scores[mask])
            blended = blend_with_rgb(overlay, rgb_base, overlay_alpha.value)
            render_panel(asset, blended)
            legend_md.content = (
                f"prompt=`{text}` • thr={threshold.value:.3f} • alpha={overlay_alpha.value:.2f}\n\n"
                f"score range: [{scores.min():.3f}, {scores.max():.3f}], "
                f"matched {int(mask.sum())}/{len(scores)} ({100*mask.mean():.1f}%)"
            )
            status_md.content = "_rendered single-query heatmap_"
            return

        if m == "cluster":
            tf = text_encoder.encode([text])[0]
            scores = query_single(asset, tf)

            # Heatmap base: same blend as single-query mode, using percentile cutoff
            # so the colored points line up with what the clustering sees.
            params = ClusterParams(
                top_percentile=float(cluster_top_pct.value),
                threshold=None,
                eps=float(cluster_eps.value),
                min_points=40,
                top_k=int(cluster_top_k.value),
            )
            cutoff = float(np.percentile(scores, params.top_percentile))
            mask = scores >= cutoff
            overlay = rgb_base.copy()
            if mask.any():
                overlay[mask] = heatmap_colors(scores[mask])
            blended = blend_with_rgb(overlay, rgb_base, overlay_alpha.value)
            render_panel(asset, blended)

            # Cluster on the centered display coords so marker positions line up.
            result = candidates_from_heatmap(asset.coord, scores, params)
            cands = result["candidates"]

            clear_cluster_markers()
            handles = []
            radius = float(cluster_marker_size.value)
            palette = cluster_palette(max(len(cands), 1))
            for c in cands:
                rank = c["rank"]
                center = tuple(float(v) for v in c["center"])
                color = tuple(int(v) for v in palette[rank])
                handles.append(
                    server.scene.add_icosphere(
                        f"/cluster/sphere_{rank}",
                        radius=radius,
                        color=color,
                        opacity=0.85,
                        position=center,
                    )
                )
                bb_min = np.asarray(c["bbox_min"], dtype=np.float32)
                bb_max = np.asarray(c["bbox_max"], dtype=np.float32)
                segs = _bbox_edge_segments(bb_min, bb_max)
                handles.append(
                    server.scene.add_line_segments(
                        f"/cluster/bbox_{rank}",
                        points=segs,
                        colors=color,  # (3,) → single color across all 12 edges
                        line_width=2.0,
                    )
                )
                label_pos = (center[0], center[1], float(bb_max[2]) + 0.15)
                handles.append(
                    server.scene.add_label(
                        f"/cluster/label_{rank}",
                        text=f"#{rank} n={c['n_points']} s={c['mean_score']:.3f}",
                        position=label_pos,
                    )
                )
            state["cluster_handles"] = handles

            legend_lines = [
                f"prompt=`{text}` • top-pct={params.top_percentile:.1f} "
                f"• eps={params.eps:.2f} • top-K={params.top_k}",
                f"score range [{result['stats']['score_min']:.3f}, {result['stats']['score_max']:.3f}] • "
                f"active {result['stats']['n_active_points']} pts • "
                f"clusters={result['stats']['n_clusters']} (noise {result['stats']['n_noise_points']})",
                "",
            ]
            if not cands:
                legend_lines.append("_no candidates — try a lower top-percentile or smaller eps_")
            for c in cands:
                col = palette[c["rank"]]
                legend_lines.append(
                    f"- <span style='color: rgb({col[0]},{col[1]},{col[2]})'>●</span> "
                    f"**#{c['rank']}** center=({c['center'][0]:.2f}, {c['center'][1]:.2f}, "
                    f"{c['center'][2]:.2f}) • n={c['n_points']} • "
                    f"mean={c['mean_score']:.3f} • max={c['max_score']:.3f}"
                )
            legend_md.content = "\n".join(legend_lines)
            status_md.content = (
                f"_clustered: {len(cands)} candidates in {result['stats']['elapsed_s']:.2f}s_"
            )
            return

        if m == "class list":
            classes = [c.strip() for c in text.split(",") if c.strip()]
            if not classes:
                status_md.content = "_no classes_"
                return
            tf = text_encoder.encode(classes)
            labels, top_scores = query_classes(asset, tf)
            palette = class_colors(len(classes))
            overlay = palette[labels]
            low = top_scores < (top_scores.max() - 0.2)
            overlay[low] = (overlay[low] * 0.5).astype(np.uint8)
            blended = blend_with_rgb(overlay, rgb_base, overlay_alpha.value)
            render_panel(asset, blended)

            counts = np.bincount(labels, minlength=len(classes))
            legend_lines = [f"class mode • k={len(classes)}"]
            for i, c in enumerate(classes):
                col = palette[i]
                legend_lines.append(
                    f"- <span style='color: rgb({col[0]},{col[1]},{col[2]})'>■</span> "
                    f"{c}: {counts[i]} ({100*counts[i]/len(labels):.1f}%)"
                )
            legend_md.content = "\n".join(legend_lines)
            status_md.content = "_rendered class-argmax_"
            return

    @view_level.on_update
    def _(_):
        is_building = view_level.value == "building"
        building_dropdown.visible = is_building
        region_dropdown.visible = not is_building
        highlight_dropdown.visible = is_building
        show_region_labels.visible = is_building
        if is_building:
            load_and_render_building(building_dropdown.value)
        else:
            load_and_render(region_dropdown.value)

    @highlight_dropdown.on_update
    def _(_):
        asset: RegionAssets = state["asset"]  # type: ignore[assignment]
        if asset is None or asset.region_slices is None:
            return
        # Re-render with whatever query/mode is currently active so the highlight
        # composes on top of (e.g.) heatmaps too.
        apply_query()

    @show_region_labels.on_update
    def _(_):
        asset: RegionAssets = state["asset"]  # type: ignore[assignment]
        if asset is None:
            return
        if view_level.value == "building" and show_region_labels.value:
            render_region_labels(asset)
        else:
            clear_region_labels()

    @building_dropdown.on_update
    def _(_):
        if view_level.value == "building":
            load_and_render_building(building_dropdown.value)

    @region_dropdown.on_update
    def _(_):
        if view_level.value == "room":
            load_and_render(region_dropdown.value)

    @apply_btn.on_click
    def _(_):
        try:
            apply_query()
        except Exception as e:  # surface errors in UI
            status_md.content = f"_error: {e}_"
            raise

    @point_size_slider.on_update
    def _(_):
        if state["asset"] is not None:
            apply_query()

    @overlay_alpha.on_update
    def _(_):
        if state["asset"] is not None:
            apply_query()

    # initial load — building view by default
    load_and_render_building(buildings[0])

    print(f"[viser] running at http://{args.host}:{args.port}")
    print("[viser] Ctrl+C to exit")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
