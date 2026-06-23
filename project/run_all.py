"""
data/npy/ 아래 모든 방에 대해 infer 실행.
OOM 발생 시 자동으로 2-block crop 모드로 fallback.
모델은 한 번만 로드.
"""
import sys
sys.path.insert(0, '/shareHost/minyoy/unidet3d')

from unidet3d.data_preprocessor import Det3DDataPreprocessor_
from unidet3d.unidet3d import UniDet3D

import os
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, PointData

# ============================================================
# 설정
# ============================================================
CFG  = '/shareHost/minyoy/unidet3d/configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py'
CKPT = '/shareHost/minyoy/unidet3d/work_dirs/unidet3d.pth'

NPY_ROOT = '/shareHost/minyoy/project/data/npy'
OUT_ROOT = '/shareHost/minyoy/project/data/npy'  # detections.pkl을 npy 폴더에 저장

DATASET_NAME = 'multiscan'

# 일반 모드 설정 (infer.py 기준)
VOXEL_DOWNSAMPLE = 0.02
MAX_POINTS       = 80000
SUPERPOINT_VOXEL = 0.10
SCORE_THR        = 0.30

# crop 모드 설정 (infer_crop.py 기준)
CROP_MIN_POINTS  = 800
CROP_OVERLAP     = 0.7
CROP_DUP_DIST    = 0.8
CROP_SCORE_THR   = 0.90

MULTISCAN_CLASSES = [
    'door', 'table', 'chair', 'cabinet', 'window', 'sofa', 'microwave',
    'pillow', 'tv_monitor', 'curtain', 'trash_can', 'suitcase', 'sink',
    'backpack', 'bed', 'refrigerator', 'toilet',
]

CLASS_NAMES = MULTISCAN_CLASSES


# ============================================================
# 유틸
# ============================================================
def load_points(npy_dir: Path) -> np.ndarray:
    """npy → (N, 9) float32 array. convert.py와 동일한 전처리."""
    coord  = np.load(npy_dir / 'coord.npy').astype(np.float32)
    color  = np.load(npy_dir / 'color.npy').astype(np.float32)
    normal = np.load(npy_dir / 'normal.npy').astype(np.float32)

    if color.max() > 1.0:
        color = color / 127.5 - 1.0

    voxel_idx = np.floor(coord / VOXEL_DOWNSAMPLE).astype(np.int64)
    _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)

    coord  = coord[unique_idx]
    color  = color[unique_idx]
    normal = normal[unique_idx]

    return np.concatenate([coord, color, normal], axis=1).astype(np.float32)


def make_voxel_superpoints(coords, voxel_size=SUPERPOINT_VOXEL):
    xyz_min = coords.min(axis=0, keepdims=True)
    voxel_coord = np.floor((coords - xyz_min) / voxel_size).astype(np.int32)
    _, inverse = np.unique(voxel_coord, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def random_sample(pts, max_points):
    n = len(pts)
    if n <= max_points:
        return pts, np.arange(n)
    choice = np.sort(np.random.choice(n, max_points, replace=False))
    return pts[choice], choice


def build_data_sample(sp):
    ds = Det3DDataSample()
    ds.set_metainfo({
        'lidar_path': f'{DATASET_NAME}/my_scene.bin',
        'pts_filename': f'{DATASET_NAME}/my_scene.bin',
        'file_name': f'{DATASET_NAME}/my_scene.bin',
        'box_type_3d': 'Depth',
        'box_mode_3d': 2,
    })
    gt_pts_seg = PointData()
    gt_pts_seg.sp_pts_mask = sp
    ds.gt_pts_seg = gt_pts_seg
    return ds


def predict_block(model, block_pts):
    """block_pts: (N, 9) numpy. (bboxes, scores, labels) 반환."""
    pts_s, _ = random_sample(block_pts, MAX_POINTS)
    sp_np = make_voxel_superpoints(pts_s[:, :3])

    pts_t = torch.from_numpy(pts_s).float().cuda()
    sp_t  = torch.from_numpy(sp_np).long().cuda()

    ds = build_data_sample(sp_t)
    batch = dict(points=[pts_t], sp_pts_masks=[sp_t])

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            results = model.predict(batch, [ds])

    r = results[0]
    bboxes = r.pred_instances_3d.bboxes_3d.tensor.detach().cpu().numpy()
    scores = r.pred_instances_3d.scores_3d.detach().cpu().numpy()
    labels = r.pred_instances_3d.labels_3d.detach().cpu().numpy()

    del pts_t, sp_t, batch, results
    torch.cuda.empty_cache()

    return bboxes, scores, labels


def split_two_blocks(pts, min_points=CROP_MIN_POINTS, overlap=CROP_OVERLAP):
    coords = pts[:, :3]
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()

    if x_range >= y_range:
        axis = 0
        lo, hi = coords[:, 0].min(), coords[:, 0].max()
    else:
        axis = 1
        lo, hi = coords[:, 1].min(), coords[:, 1].max()

    mid = (lo + hi) / 2.0
    c = coords[:, axis]

    idx0 = np.where((c >= lo) & (c <= mid + overlap))[0]
    idx1 = np.where((c >= mid - overlap) & (c <= hi))[0]

    blocks = []
    for idx in [idx0, idx1]:
        if len(idx) >= min_points:
            blocks.append(idx)
    return blocks


def remove_duplicates(bboxes, scores, labels, dist_thr=CROP_DUP_DIST):
    if len(bboxes) == 0:
        return bboxes, scores, labels
    order = np.argsort(scores)[::-1]
    centers = bboxes[:, :3]
    keep = []
    for i in order:
        dup = any(
            int(labels[i]) == int(labels[j]) and
            np.linalg.norm(centers[i] - centers[j]) < dist_thr
            for j in keep
        )
        if not dup:
            keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    return bboxes[keep], scores[keep], labels[keep]


def get_box_points(coords, bboxes):
    box_pts = []
    for box in bboxes:
        cx, cy, cz, dx, dy, dz = box[:6]
        inbox = (
            (coords[:, 0] > cx - dx / 2) & (coords[:, 0] < cx + dx / 2) &
            (coords[:, 1] > cy - dy / 2) & (coords[:, 1] < cy + dy / 2) &
            (coords[:, 2] > cz - dz / 2) & (coords[:, 2] < cz + dz / 2)
        )
        box_pts.append(np.where(inbox)[0])
    return box_pts


def save_detections(out_path, pts_original, bboxes, scores, labels, mode):
    box_pts = get_box_points(pts_original[:, :3], bboxes)
    detections = dict(
        dataset_name=DATASET_NAME,
        points=pts_original,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        box_pts=box_pts,
        classes=CLASS_NAMES,
        infer_mode=mode,
    )
    with open(out_path, 'wb') as f:
        pickle.dump(detections, f)


# ============================================================
# 방 하나 처리
# ============================================================
def process_room(model, room_dir: Path):
    out_path = room_dir / 'detections.pkl'

    if out_path.exists():
        print(f'[SKIP] {room_dir.name} — detections.pkl 이미 존재')
        return

    print(f'\n{"="*60}')
    print(f'[ROOM] {room_dir.name}')

    pts = load_points(room_dir)
    print(f'  포인트 수 (다운샘플 후): {len(pts)}')

    # ── 일반 모드 ──────────────────────────────────────────
    try:
        bboxes, scores, labels = predict_block(model, pts)

        mask = scores > SCORE_THR
        bboxes, scores, labels = bboxes[mask], scores[mask], labels[mask]

        print(f'  [normal] bbox: {len(bboxes)}개')
        save_detections(out_path, pts, bboxes, scores, labels, mode='normal')
        print(f'  저장 → {out_path}')
        return

    except torch.cuda.OutOfMemoryError:
        print('  [OOM] 일반 모드 실패 → crop 모드로 재시도')
        torch.cuda.empty_cache()

    # ── crop 모드 ──────────────────────────────────────────
    block_indices = split_two_blocks(pts)
    print(f'  crop 블록 수: {len(block_indices)}')

    all_bboxes, all_scores, all_labels = [], [], []

    for i, idx in enumerate(block_indices):
        block_pts = pts[idx]
        print(f'  [block {i}] {len(block_pts)} points')
        try:
            bboxes, scores, labels = predict_block(model, block_pts)

            mask = scores > CROP_SCORE_THR
            bboxes, scores, labels = bboxes[mask], scores[mask], labels[mask]

            print(f'    bbox (thr): {len(bboxes)}개')
            if len(bboxes) > 0:
                all_bboxes.append(bboxes)
                all_scores.append(scores)
                all_labels.append(labels)

        except torch.cuda.OutOfMemoryError:
            print(f'    [OOM] block {i} 건너뜀')
            torch.cuda.empty_cache()

    if len(all_bboxes) == 0:
        print('  검출 결과 없음 — 빈 detections 저장')
        bboxes = np.zeros((0, 7), dtype=np.float32)
        scores = np.zeros((0,), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int64)
    else:
        bboxes = np.concatenate(all_bboxes)
        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        bboxes, scores, labels = remove_duplicates(bboxes, scores, labels)

    print(f'  [crop] 최종 bbox: {len(bboxes)}개')
    save_detections(out_path, pts, bboxes, scores, labels, mode='crop')
    print(f'  저장 → {out_path}')


# ============================================================
# 메인
# ============================================================
def main():
    cfg = Config.fromfile(CFG)

    from mmengine.registry import MODELS as MMENGINE_MODELS
    if 'Det3DDataPreprocessor_' not in MMENGINE_MODELS._module_dict:
        MMENGINE_MODELS.register_module(module=Det3DDataPreprocessor_)

    print('모델 로딩 중...')
    model = MODELS.build(cfg.model)
    load_checkpoint(model, CKPT, map_location='cpu')
    model = model.cuda()
    model.eval()
    print('모델 로드 완료')

    if DATASET_NAME not in model.decoder.datasets:
        raise ValueError(f'DATASET_NAME={DATASET_NAME} not in {model.decoder.datasets}')

    rooms = sorted([
        p for p in Path(NPY_ROOT).iterdir()
        if p.is_dir() and (p / 'coord.npy').exists()
    ])

    print(f'\n총 {len(rooms)}개 방 처리 시작')

    for room_dir in rooms:
        try:
            process_room(model, room_dir)
        except Exception as e:
            print(f'  [ERROR] {room_dir.name}: {e}')

    print('\n\n모든 방 처리 완료')


if __name__ == '__main__':
    main()
