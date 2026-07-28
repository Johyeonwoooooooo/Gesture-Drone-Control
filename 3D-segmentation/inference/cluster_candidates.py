"""Cluster the per-point CLIP heatmap into ranked 3D object candidates.

Pipeline (matches the data layout produced by ``run_inference.py``):

    1. Load ``cache/feat/<region>/{feat.npy, coord.npy}``.
       feat is (N, D) fp16 (already CLIP-text-aligned), coord is (N, 3) float32
       in original world coordinates.
    2. Encode the natural-language prompt with the same CLIP text encoder used by
       the webapp (``UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP``).
    3. Cosine-sim per point -> heatmap scores.
    4. Threshold by top-percentile (default) or fixed cutoff -> "active" subset.
    5. Spatial clustering (DBSCAN) on the active point coordinates.
    6. Aggregate per cluster: centroid, bbox, mean/max score, point count.
    7. Rank by ``mean_score * sqrt(n_points)`` and keep top-K candidates.

The output is a JSON document. Each candidate is one putative object instance
in world coordinates and is intended to be consumed by the downstream path
planner (one target = one cluster centroid).

CLI usage:

    python inference/cluster_candidates.py \
        --region 17DRP5sb8fy_00 --query "a tv" \
        --top-percentile 95 --eps 0.25 --min-points 40 --top-k 5

Programmatic usage:

    from inference.cluster_candidates import extract_candidates
    result = extract_candidates(region="17DRP5sb8fy_00", query="a tv")
    for c in result["candidates"]:
        print(c["rank"], c["center"], c["mean_score"])
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sklearn.cluster import DBSCAN

CLIP_MODEL_ID = "hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP"
DEFAULT_FEAT_DIR = Path(__file__).resolve().parents[1] / "cache" / "feat"


@dataclass
class ClusterParams:
    top_percentile: Optional[float] = 95.0  # None when an absolute threshold is given
    threshold: Optional[float] = None
    eps: float = 0.25  # DBSCAN neighbourhood radius (meters)
    min_points: int = 40  # DBSCAN min_samples
    top_k: int = 5

    def validate(self) -> None:
        if self.threshold is None and self.top_percentile is None:
            raise ValueError("Provide either threshold or top_percentile")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if self.min_points < 1:
            raise ValueError("min_points must be >= 1")


def _load_clip(device: torch.device):
    """Return (text_encode_fn) for the same CLIP text head the inference cache was aligned to."""
    from open_clip import create_model_and_transforms, get_tokenizer

    print(f"[clip] loading {CLIP_MODEL_ID} ...", flush=True)
    model, _, _ = create_model_and_transforms(CLIP_MODEL_ID, device=device)
    tokenizer = get_tokenizer(CLIP_MODEL_ID)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def encode(prompt: str) -> torch.Tensor:
        tok = tokenizer([prompt]).to(device)
        feat = model.encode_text(tok)
        return torch.nn.functional.normalize(feat, dim=-1)[0]

    return encode


def _load_region(region_dir: Path, device: torch.device):
    feat = np.load(region_dir / "feat.npy")  # (N, 768) fp16
    coord = np.load(region_dir / "coord.npy").astype(np.float32, copy=True)
    feat_t = torch.from_numpy(feat).to(device, dtype=torch.float32)
    feat_t = torch.nn.functional.normalize(feat_t, dim=-1)
    return feat_t, coord


def _heatmap(feat_t: torch.Tensor, text_feat: torch.Tensor) -> np.ndarray:
    sim = (feat_t @ text_feat.unsqueeze(-1)).squeeze(-1)
    return sim.detach().cpu().numpy()


def _active_mask(scores: np.ndarray, params: ClusterParams) -> np.ndarray:
    if params.threshold is not None:
        return scores >= params.threshold
    cutoff = float(np.percentile(scores, params.top_percentile))
    return scores >= cutoff


def _cluster(coords: np.ndarray, params: ClusterParams) -> np.ndarray:
    db = DBSCAN(eps=params.eps, min_samples=params.min_points, n_jobs=-1).fit(coords)
    return db.labels_


def _summarize(
    active_coord: np.ndarray,
    active_score: np.ndarray,
    labels: np.ndarray,
    top_k: int,
) -> List[dict]:
    candidates: List[dict] = []
    for cid in np.unique(labels):
        if cid < 0:  # DBSCAN noise
            continue
        mask = labels == cid
        c_coord = active_coord[mask]
        c_score = active_score[mask]
        candidates.append(
            dict(
                cluster_id=int(cid),
                center=c_coord.mean(0).tolist(),
                bbox_min=c_coord.min(0).tolist(),
                bbox_max=c_coord.max(0).tolist(),
                n_points=int(mask.sum()),
                mean_score=float(c_score.mean()),
                max_score=float(c_score.max()),
            )
        )
    candidates.sort(key=lambda c: -(c["mean_score"] * (c["n_points"] ** 0.5)))
    candidates = candidates[:top_k]
    for i, c in enumerate(candidates):
        c["rank"] = i
    return candidates


def candidates_from_heatmap(
    coord: np.ndarray,
    scores: np.ndarray,
    params: Optional[ClusterParams] = None,
) -> dict:
    """Run threshold + DBSCAN + ranking on an already-computed per-point heatmap.

    Useful when the caller already has CLIP features and per-point scores in
    memory (e.g. the viser webapp). For the full load-from-cache pipeline use
    :func:`extract_candidates`.

    Args:
        coord: (N, 3) float — per-point coordinates in whatever frame you want
            the candidate centroids/bboxes returned in.
        scores: (N,) float — per-point cosine similarity to the target prompt.
        params: clustering / ranking parameters. ``ClusterParams()`` defaults
            apply when omitted.
    """
    params = params or ClusterParams()
    params.validate()

    if coord.shape[0] != scores.shape[0]:
        raise ValueError(f"coord/scores length mismatch: {coord.shape[0]} vs {scores.shape[0]}")

    t0 = time.time()
    mask = _active_mask(scores, params)
    n_active = int(mask.sum())

    candidates: List[dict] = []
    n_clusters = 0
    n_noise = 0
    if n_active > 0:
        labels = _cluster(coord[mask], params)
        candidates = _summarize(coord[mask], scores[mask], labels, params.top_k)
        n_clusters = int((np.unique(labels) >= 0).sum())
        n_noise = int((labels < 0).sum())

    return {
        "params": {
            "top_percentile": params.top_percentile if params.threshold is None else None,
            "threshold": params.threshold,
            "eps": params.eps,
            "min_points": params.min_points,
            "top_k": params.top_k,
        },
        "stats": {
            "n_total_points": int(coord.shape[0]),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "score_p95": float(np.percentile(scores, 95)),
            "n_active_points": n_active,
            "n_clusters": n_clusters,
            "n_noise_points": n_noise,
            "elapsed_s": round(time.time() - t0, 3),
        },
        "candidates": candidates,
    }


def extract_candidates(
    region: str,
    query: str,
    *,
    feat_dir: Path = DEFAULT_FEAT_DIR,
    params: Optional[ClusterParams] = None,
    device: Optional[str] = None,
) -> dict:
    """Run the full heatmap-clustering pipeline for one (region, query) pair.

    Loads the cached features for ``region``, encodes ``query`` with CLIP,
    computes the heatmap and delegates to :func:`candidates_from_heatmap`.
    Returns a JSON-serialisable dict with the ranked candidate list and metadata.
    """
    params = params or ClusterParams()
    params.validate()

    feat_dir = Path(feat_dir)
    region_dir = feat_dir / region
    if not (region_dir / "feat.npy").exists():
        raise FileNotFoundError(f"feat cache not found: {region_dir} (run inference first)")

    dev = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    feat_t, coord = _load_region(region_dir, dev)
    encode = _load_clip(dev)
    text_feat = encode(query)
    scores = _heatmap(feat_t, text_feat)

    out = candidates_from_heatmap(coord, scores, params)
    return {"region": region, "query": query, **out}


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--feat-dir", default=str(DEFAULT_FEAT_DIR), help="Root of cache/feat")
    ap.add_argument("--region", required=True, help="Region folder name under feat-dir")
    ap.add_argument("--query", required=True, help="Natural-language target prompt")
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--top-percentile",
        type=float,
        default=95.0,
        help="Keep points whose score is in the top X percent (default 95.0).",
    )
    g.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Absolute cosine-sim cutoff; overrides --top-percentile if given.",
    )
    ap.add_argument("--eps", type=float, default=0.25, help="DBSCAN eps in meters")
    ap.add_argument("--min-points", type=int, default=40, help="DBSCAN min_samples")
    ap.add_argument("--top-k", type=int, default=5, help="Max candidates to return")
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--out",
        default=None,
        help="Write JSON to this path. If omitted, prints to stdout.",
    )
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    params = ClusterParams(
        top_percentile=None if args.threshold is not None else args.top_percentile,
        threshold=args.threshold,
        eps=args.eps,
        min_points=args.min_points,
        top_k=args.top_k,
    )
    result = extract_candidates(
        region=args.region,
        query=args.query,
        feat_dir=Path(args.feat_dir),
        params=params,
        device=args.device,
    )
    js = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(js)
        print(f"[done] wrote {args.out} ({len(result['candidates'])} candidates)")
    else:
        print(js)


if __name__ == "__main__":
    main()
