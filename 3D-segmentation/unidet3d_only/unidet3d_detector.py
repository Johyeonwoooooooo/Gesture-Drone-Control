"""UniDet3D 3D object detector (refactor of project/infer.py).

Provides a reusable `UniDet3DDetector` class for online use inside webapp_llm.

Pipeline per scene:
    (N, 9) points (x,y,z,r,g,b,nx,ny,nz)
        -> voxel superpoint grouping
        -> UniDet3D forward
        -> filtered bboxes / labels / scores
        -> per-bbox member-point indices (against the ORIGINAL point cloud)

Heavy deps (mmengine, mmdet3d, unidet3d) are imported lazily inside the
class so that webapp_llm can start without them when UniDet3D mode is off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch


# ScanNet++ label head class names (matches project/infer.py).
SCANNETPP_CLASSES: List[str] = [
    'table', 'door', 'ceiling lamp', 'cabinet', 'blinds', 'curtain', 'chair',
    'storage cabinet', 'office chair', 'bookshelf', 'whiteboard', 'window',
    'box', 'monitor', 'shelf', 'heater', 'kitchen cabinet', 'sofa', 'bed',
    'trash can', 'book', 'plant', 'blanket', 'tv', 'computer tower',
    'refrigerator', 'jacket', 'sink', 'bag', 'picture', 'pillow', 'towel',
    'suitcase', 'backpack', 'crate', 'keyboard', 'rack', 'toilet', 'printer',
    'poster', 'painting', 'microwave', 'shoes', 'socket', 'bottle', 'bucket',
    'cushion', 'basket', 'shoe rack', 'telephone', 'file folder', 'laptop',
    'plant pot', 'exhaust fan', 'cup', 'coat hanger', 'light switch',
    'speaker', 'table lamp', 'kettle', 'smoke detector', 'container',
    'power strip', 'slippers', 'paper bag', 'mouse', 'cutting board',
    'toilet paper', 'paper towel', 'pot', 'clock', 'pan', 'tap', 'jar',
    'soap dispenser', 'binder', 'bowl', 'tissue box', 'whiteboard eraser',
    'toilet brush', 'spray bottle', 'headphones', 'stapler', 'marker',
]


@dataclass
class DetectionResult:
    bboxes: np.ndarray          # (M, 7) or (M, 6) — cx,cy,cz,dx,dy,dz[,yaw] in WORLD frame
    scores: np.ndarray          # (M,)
    labels: np.ndarray          # (M,) int class index
    classes: List[str]          # length C — class names for the active head
    box_pts: List[np.ndarray]   # per-bbox member-point indices into the ORIGINAL pts


def make_voxel_superpoints(coords: np.ndarray, voxel_size: float = 0.10) -> np.ndarray:
    xyz_min = coords.min(axis=0, keepdims=True)
    voxel_coord = np.floor((coords - xyz_min) / voxel_size).astype(np.int32)
    _, inverse = np.unique(voxel_coord, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def random_sample_points(pts: np.ndarray, max_points: int = 80000):
    n = len(pts)
    if n <= max_points:
        return pts, np.arange(n)
    choice = np.random.choice(n, max_points, replace=False)
    choice = np.sort(choice)
    return pts[choice], choice


class UniDet3DDetector:
    """Lazy-loaded UniDet3D wrapper.

    Args:
        cfg_path: mmengine config path used to train the multi-head checkpoint.
        ckpt_path: model weights.
        unidet3d_root: path to add to sys.path so that
            `from unidet3d.unidet3d import UniDet3D` resolves. Required because
            UniDet3D ships as a research repo, not a pip package.
        device: torch device for inference.
        dataset_name: which decoder head to use (e.g. 'scannetpp').
    """

    def __init__(
        self,
        cfg_path: str,
        ckpt_path: str,
        unidet3d_root: str,
        device: str = 'cuda',
        dataset_name: str = 'scannetpp',
    ):
        self.cfg_path = cfg_path
        self.ckpt_path = ckpt_path
        self.unidet3d_root = unidet3d_root
        self.device = device
        self.dataset_name = dataset_name

        self._model = None
        self._classes: Optional[List[str]] = None

    # ----- internal -----
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import sys
        if self.unidet3d_root not in sys.path:
            sys.path.insert(0, self.unidet3d_root)

        # Must import these so UniDet3D registers its modules BEFORE mmdet3d builds.
        from unidet3d.data_preprocessor import Det3DDataPreprocessor_  # noqa: F401
        from unidet3d.unidet3d import UniDet3D  # noqa: F401

        from mmengine.config import Config
        from mmengine.runner.checkpoint import _load_checkpoint
        from mmengine.registry import MODELS as MMENGINE_MODELS
        from mmdet3d.registry import MODELS

        cfg = Config.fromfile(self.cfg_path)

        if 'Det3DDataPreprocessor_' not in MMENGINE_MODELS._module_dict:
            MMENGINE_MODELS.register_module(module=Det3DDataPreprocessor_)

        model = MODELS.build(cfg.model)
        self._load_clean_checkpoint(model, _load_checkpoint)
        model = model.to(self.device).eval()

        if self.dataset_name not in model.decoder.datasets:
            raise ValueError(
                f'dataset_name={self.dataset_name} not in decoder heads '
                f'{model.decoder.datasets}'
            )

        if self.dataset_name == 'scannetpp':
            classes = SCANNETPP_CLASSES
        else:
            classes = self._resolve_classes(cfg)

        self._model = model
        self._classes = classes
        print(f'[unidet3d] loaded {self.dataset_name} head '
              f'({len(classes)} classes) on {self.device}')

    def _load_clean_checkpoint(self, model, _load_checkpoint) -> None:
        """Load the checkpoint after stripping vestigial positional-encoding keys.

        The released ``unidet3d.pth`` carries 12 unused
        ``decoder.self_attn_layers.*.pe.1.*`` tensors from an earlier training
        architecture. Upstream's published ``encoder.py`` has no ``pe`` module,
        so a plain ``load_checkpoint`` succeeds but prints a noisy
        "model and loaded state dict do not match exactly / unexpected key ...
        pe.1 ..." warning. Those keys load nothing and change no behaviour
        (``missing_keys`` is empty); we drop them so the load is clean and a
        *real* future mismatch isn't lost in the noise.
        """
        ckpt = _load_checkpoint(self.ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        dropped = [k for k in state if '.pe.' in k]
        if dropped:
            for k in dropped:
                del state[k]
            print(f'[unidet3d] dropped {len(dropped)} vestigial pe.* '
                  f'checkpoint keys before load')
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f'[unidet3d] WARNING state_dict still mismatched after pe strip '
                  f'— missing={list(missing)} unexpected={list(unexpected)}')

    def _resolve_classes(self, cfg) -> List[str]:
        datasets = cfg.test_dataloader.dataset.datasets
        order = ['scannet', 's3dis', 'multiscan', '3rscan', 'scannetpp', 'arkitscenes']
        if self.dataset_name in order:
            idx = order.index(self.dataset_name)
            if idx < len(datasets):
                meta = datasets[idx].get('metainfo', {})
                cls = meta.get('classes', None)
                if cls:
                    return list(cls)
        return [f'{self.dataset_name}_class_{i}' for i in range(200)]

    # ----- public -----
    @property
    def classes(self) -> List[str]:
        self._ensure_loaded()
        return list(self._classes)  # type: ignore[arg-type]

    @torch.no_grad()
    def detect(
        self,
        pts: np.ndarray,
        score_thr: float = 0.30,
        max_points: int = 80000,
        voxel_size: float = 0.10,
    ) -> DetectionResult:
        """Run UniDet3D on a (N, 9) point cloud.

        `pts` is expected in WORLD coordinates with channels
        (x, y, z, r, g, b, nx, ny, nz). Coordinates are not modified.
        """
        self._ensure_loaded()
        if pts.ndim != 2 or pts.shape[1] not in (6, 9):
            raise ValueError(f'expected (N, 6) xyz+rgb or (N, 9) xyz+rgb+normal, got {pts.shape}')
        if pts.shape[1] == 9:
            pts = pts[:, :6]  # UniDet3D only uses xyz+rgb (see configs: use_dim=[0..5])

        coords_original = pts[:, :3]
        pts_s, sampled_idx = random_sample_points(pts, max_points=max_points)
        coords = pts_s[:, :3]

        sp_np = make_voxel_superpoints(coords, voxel_size=voxel_size)

        pts_tensor = torch.from_numpy(pts_s).float().to(self.device)
        sp = torch.from_numpy(sp_np).long().to(self.device)

        from mmdet3d.structures import Det3DDataSample, PointData
        ds = Det3DDataSample()
        ds.set_metainfo({
            'lidar_path': f'{self.dataset_name}/scene.bin',
            'pts_filename': f'{self.dataset_name}/scene.bin',
            'file_name': f'{self.dataset_name}/scene.bin',
            'box_type_3d': 'Depth',
            'box_mode_3d': 2,
        })
        gt = PointData()
        gt.sp_pts_mask = sp
        ds.gt_pts_seg = gt

        batch_inputs = dict(points=[pts_tensor], sp_pts_masks=[sp])

        with torch.amp.autocast('cuda'):
            results = self._model.predict(batch_inputs, [ds])  # type: ignore[union-attr]

        r = results[0].pred_instances_3d
        bboxes = r.bboxes_3d.tensor.detach().cpu().numpy()
        scores = r.scores_3d.detach().cpu().numpy()
        labels = r.labels_3d.detach().cpu().numpy()

        keep = scores > score_thr
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # member-point indices against the ORIGINAL pts
        box_pts: List[np.ndarray] = []
        for box in bboxes:
            cx, cy, cz, dx, dy, dz = box[:6]
            inbox = (
                (coords_original[:, 0] > cx - dx / 2) &
                (coords_original[:, 0] < cx + dx / 2) &
                (coords_original[:, 1] > cy - dy / 2) &
                (coords_original[:, 1] < cy + dy / 2) &
                (coords_original[:, 2] > cz - dz / 2) &
                (coords_original[:, 2] < cz + dz / 2)
            )
            box_pts.append(np.where(inbox)[0])

        return DetectionResult(
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            classes=list(self._classes),  # type: ignore[arg-type]
            box_pts=box_pts,
        )


def build_class_embeds(
    classes: Sequence[str],
    text_encoder,  # webapp.server.TextEncoder
    prompt_template: str = 'a photo of a {}',
) -> torch.Tensor:
    """Encode each class name with the shared CLIP TextEncoder.

    Returns a (C, D) tensor, L2-normalized.
    """
    prompts = [prompt_template.format(c) for c in classes]
    feats = text_encoder.encode(prompts)  # already normalized
    return feats


def topk_boxes_for_query(
    query_feat: torch.Tensor,            # (D,) normalized
    box_class_embeds: torch.Tensor,      # (M, D) — class embedding tiled per box
    topk: int,
):
    """Return (top_idx[k], sims[M]) sorted by cosine similarity."""
    sims = (box_class_embeds @ query_feat.view(-1, 1)).squeeze(-1)
    sims_np = sims.detach().cpu().numpy()
    order = np.argsort(sims_np)[::-1][:topk]
    return order, sims_np
