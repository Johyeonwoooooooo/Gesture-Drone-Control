# infer_crop.py
import sys
sys.path.insert(0, '/shareHost/minyoy/unidet3d')

# mmdet3d보다 먼저 unidet3d 등록
from unidet3d.data_preprocessor import Det3DDataPreprocessor_
from unidet3d.unidet3d import UniDet3D

import numpy as np
import pickle
import torch
from collections import Counter
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS


CFG = '/shareHost/minyoy/unidet3d/configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py'
CKPT = '/shareHost/minyoy/unidet3d/work_dirs/unidet3d.pth'

INPUT = '/shareHost/minyoy/project/data/00809_Qpor2mEya8F_000_002/my_scene.bin'
OUT = '/shareHost/minyoy/project/data/00809_Qpor2mEya8F_000_002/detections.pkl'

# candidates: scannet, s3dis, multiscan, 3rscan, scannetpp, arkitscenes
DATASET_NAME = 'multiscan'

# coord + color + normal이면 9
# coord + color만 쓰면 6
POINT_DIM = 9

# 2-block crop inference 설정
MIN_POINTS = 800
MAX_POINTS_PER_BLOCK = 80000
SPLIT_OVERLAP = 0.7

# superpoint 설정
SUPERPOINT_VOXEL_SIZE = 0.10

# score threshold
SCORE_THR = 0.90

# 중복 bbox 제거 설정
DUP_CENTER_DIST_THR = 0.8


MULTISCAN_CLASSES = [
    'door',
    'table',
    'chair',
    'cabinet',
    'window',
    'sofa',
    'microwave',
    'pillow',
    'tv_monitor',
    'curtain',
    'trash_can',
    'suitcase',
    'sink',
    'backpack',
    'bed',
    'refrigerator',
    'toilet',
]


SCANNET_CLASSES = [
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'desk', 'curtain',
    'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
    'otherfurniture',
]


def get_dataset_classes(cfg, dataset_name):
    if dataset_name == 'multiscan':
        return MULTISCAN_CLASSES

    if dataset_name == 'scannet':
        return SCANNET_CLASSES

    datasets = cfg.test_dataloader.dataset.datasets

    for d in datasets:
        metainfo = d.get('metainfo', {})
        classes = metainfo.get('classes', None)

        d_type = str(d.get('type', '')).lower()
        data_root = str(d.get('data_root', '')).lower()
        ann_file = str(d.get('ann_file', '')).lower()

        text = f'{d_type} {data_root} {ann_file}'

        if dataset_name.lower() in text and classes is not None and len(classes) > 0:
            return list(classes)

    raise ValueError(f'{dataset_name} classes를 찾지 못했습니다.')


def make_voxel_superpoints(coords, voxel_size=0.10):
    """
    각 점을 개별 superpoint로 두지 않고,
    voxel grid 단위로 superpoint id를 부여한다.
    """
    xyz_min = coords.min(axis=0, keepdims=True)
    voxel_coord = np.floor((coords - xyz_min) / voxel_size).astype(np.int32)

    _, inverse = np.unique(voxel_coord, axis=0, return_inverse=True)

    return inverse.astype(np.int64)


def random_sample_points(pts, max_points):
    """
    block 안의 point 수가 너무 많으면 random sampling.
    """
    n = len(pts)

    if n <= max_points:
        return pts, np.arange(n)

    choice = np.random.choice(n, max_points, replace=False)
    choice = np.sort(choice)

    return pts[choice], choice


def split_into_two_blocks(pts, min_points=800, overlap=0.7):
    """
    scene을 XY 기준으로 2개 block으로만 나눈다.
    x/y 중 더 긴 축을 기준으로 반으로 자르고, overlap을 둔다.
    """
    coords = pts[:, :3]

    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    blocks = []

    if x_range >= y_range:
        mid = (x_min + x_max) / 2.0

        mask_0 = (
            (coords[:, 0] >= x_min) &
            (coords[:, 0] <= mid + overlap)
        )

        mask_1 = (
            (coords[:, 0] >= mid - overlap) &
            (coords[:, 0] <= x_max)
        )

        idx_0 = np.where(mask_0)[0]
        idx_1 = np.where(mask_1)[0]

        print('split axis: x')
        print(f'x range: {x_min:.2f} ~ {x_max:.2f}')
        print(f'mid: {mid:.2f}, overlap: {overlap}')

    else:
        mid = (y_min + y_max) / 2.0

        mask_0 = (
            (coords[:, 1] >= y_min) &
            (coords[:, 1] <= mid + overlap)
        )

        mask_1 = (
            (coords[:, 1] >= mid - overlap) &
            (coords[:, 1] <= y_max)
        )

        idx_0 = np.where(mask_0)[0]
        idx_1 = np.where(mask_1)[0]

        print('split axis: y')
        print(f'y range: {y_min:.2f} ~ {y_max:.2f}')
        print(f'mid: {mid:.2f}, overlap: {overlap}')

    if len(idx_0) >= min_points:
        blocks.append(idx_0)
    else:
        print(f'block 0 skipped: point 수 {len(idx_0)} < {min_points}')

    if len(idx_1) >= min_points:
        blocks.append(idx_1)
    else:
        print(f'block 1 skipped: point 수 {len(idx_1)} < {min_points}')

    for i, idx in enumerate(blocks):
        print(f'block {i}: point 수 = {len(idx)}')

    return blocks


def remove_duplicate_boxes(bboxes, scores, labels, dist_thr=0.50):
    """
    두 block overlap 때문에 생기는 중복 bbox 제거.
    같은 label이고 중심점 거리가 가까우면 score 높은 것만 유지한다.
    """
    if len(bboxes) == 0:
        return bboxes, scores, labels

    order = np.argsort(scores)[::-1]
    centers = bboxes[:, :3]

    keep = []

    for i in order:
        duplicate = False

        for j in keep:
            same_label = int(labels[i]) == int(labels[j])
            close = np.linalg.norm(centers[i] - centers[j]) < dist_thr

            if same_label and close:
                duplicate = True
                break

        if not duplicate:
            keep.append(i)

    keep = np.array(keep, dtype=np.int64)

    return bboxes[keep], scores[keep], labels[keep]


def get_box_points(coords_original, bboxes):
    """
    각 bbox 내부에 들어가는 원본 point index 계산.
    app.py, build_clip_index.py에서 사용.
    """
    box_pts = []

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

    return box_pts


def build_data_sample(sp, dataset_name):
    from mmdet3d.structures import Det3DDataSample, PointData

    data_sample = Det3DDataSample()

    # 중요:
    # UniDet3D 내부 get_dataset()이 dataset 이름을 추론할 수 있도록
    # lidar_path 문자열 안에 dataset_name을 넣는다.
    data_sample.set_metainfo({
        'lidar_path': f'{dataset_name}/my_scene.bin',
        'pts_filename': f'{dataset_name}/my_scene.bin',
        'file_name': f'{dataset_name}/my_scene.bin',
        'box_type_3d': 'Depth',
        'box_mode_3d': 2,
    })

    gt_pts_seg = PointData()
    gt_pts_seg.sp_pts_mask = sp
    data_sample.gt_pts_seg = gt_pts_seg

    return data_sample


def predict_one_block(model, block_pts, dataset_name):
    """
    block 하나에 대해 UniDet3D inference 수행.
    block_pts 좌표는 원본 scene 좌표계를 유지한다.
    """
    if len(block_pts) == 0:
        return None

    block_pts_sampled, _ = random_sample_points(
        block_pts,
        max_points=MAX_POINTS_PER_BLOCK,
    )

    coords = block_pts_sampled[:, :3]

    sp_np = make_voxel_superpoints(
        coords,
        voxel_size=SUPERPOINT_VOXEL_SIZE,
    )

    num_sp = len(np.unique(sp_np))

    pts_tensor = torch.from_numpy(block_pts_sampled).float().cuda()
    sp = torch.from_numpy(sp_np).long().cuda()

    data_sample = build_data_sample(sp, dataset_name)

    batch_inputs = dict(
        points=[pts_tensor],
        sp_pts_masks=[sp],
    )

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            results = model.predict(batch_inputs, [data_sample])

    result = results[0]

    bboxes = result.pred_instances_3d.bboxes_3d.tensor.detach().cpu().numpy()
    scores = result.pred_instances_3d.scores_3d.detach().cpu().numpy()
    labels = result.pred_instances_3d.labels_3d.detach().cpu().numpy()

    del pts_tensor, sp, batch_inputs, results
    torch.cuda.empty_cache()

    return bboxes, scores, labels, num_sp, len(block_pts_sampled)


def main():
    cfg = Config.fromfile(CFG)

    from mmengine.registry import MODELS as MMENGINE_MODELS

    if 'Det3DDataPreprocessor_' not in MMENGINE_MODELS._module_dict:
        MMENGINE_MODELS.register_module(module=Det3DDataPreprocessor_)

    model = MODELS.build(cfg.model)
    load_checkpoint(model, CKPT, map_location='cpu')

    model = model.cuda()
    model.eval()

    print('decoder datasets:', model.decoder.datasets)

    if DATASET_NAME not in model.decoder.datasets:
        raise ValueError(
            f'DATASET_NAME={DATASET_NAME} 이 decoder datasets 안에 없습니다. '
            f'가능한 값: {model.decoder.datasets}'
        )

    CLASS_NAMES = get_dataset_classes(cfg, DATASET_NAME)
    print(f'{DATASET_NAME} 클래스 수: {len(CLASS_NAMES)}')

    # =========================
    # 1. point cloud 로드
    # =========================
    pts_original = np.fromfile(INPUT, dtype=np.float32).reshape(-1, POINT_DIM)
    coords_original = pts_original[:, :3]

    print(f'원본 point 수: {len(pts_original)}')
    print(f'point dim: {POINT_DIM}')

    # =========================
    # 2. scene을 2개 block으로 나누기
    # =========================
    block_indices = split_into_two_blocks(
        pts_original,
        min_points=MIN_POINTS,
        overlap=SPLIT_OVERLAP,
    )

    print(f'block 수: {len(block_indices)}')
    print(f'min_points: {MIN_POINTS}')
    print(f'split overlap: {SPLIT_OVERLAP}')
    print(f'superpoint voxel size: {SUPERPOINT_VOXEL_SIZE}')
    print(f'max points per block: {MAX_POINTS_PER_BLOCK}')

    # =========================
    # 3. block별 inference
    # =========================
    all_bboxes = []
    all_scores = []
    all_labels = []

    for block_id, idx in enumerate(block_indices):
        block_pts = pts_original[idx]

        print(f'\n[{block_id + 1}/{len(block_indices)}] block point 수: {len(block_pts)}')

        try:
            out = predict_one_block(
                model,
                block_pts,
                DATASET_NAME,
            )

            if out is None:
                print('  skip: empty block')
                continue

            bboxes, scores, labels, num_sp, sampled_n = out

            print(f'  sampled point 수: {sampled_n}')
            print(f'  superpoint 수: {num_sp}')
            print(f'  raw bbox 수: {len(bboxes)}')

            mask = scores > SCORE_THR

            bboxes = bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            print(f'  threshold 후 bbox 수: {len(bboxes)}')

            if len(bboxes) > 0:
                all_bboxes.append(bboxes)
                all_scores.append(scores)
                all_labels.append(labels)

        except torch.cuda.OutOfMemoryError:
            print('  [OOM] 이 block은 건너뜁니다.')
            torch.cuda.empty_cache()
            continue

    # =========================
    # 4. 결과 합치기
    # =========================
    if len(all_bboxes) == 0:
        print('검출된 bbox가 없습니다.')

        detections = dict(
            dataset_name=DATASET_NAME,
            points=pts_original,
            bboxes=np.zeros((0, 7), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
            box_pts=[],
            classes=CLASS_NAMES,
        )

        with open(OUT, 'wb') as f:
            pickle.dump(detections, f)

        print(f'완료 → {OUT}')
        return

    bboxes = np.concatenate(all_bboxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print('\n전체 block 합산 결과')
    print(f'중복 제거 전 bbox 수: {len(bboxes)}')

    # =========================
    # 5. 중복 bbox 제거
    # =========================
    bboxes, scores, labels = remove_duplicate_boxes(
        bboxes,
        scores,
        labels,
        dist_thr=DUP_CENTER_DIST_THR,
    )

    print(f'중복 제거 후 bbox 수: {len(bboxes)}')

    # =========================
    # 6. label 분포 출력
    # =========================
    print('\nlabel distribution:')
    cnt = Counter(labels.astype(int).tolist())

    for label_idx, count in cnt.most_common():
        if 0 <= label_idx < len(CLASS_NAMES):
            name = CLASS_NAMES[label_idx]
        else:
            name = f'unknown_{label_idx}'

        print(f'  {label_idx:3d} | {name:25s} | {count}')

    print('\ntop predictions:')
    order = np.argsort(scores)[::-1]

    for rank, i in enumerate(order[:20]):
        label_idx = int(labels[i])

        if 0 <= label_idx < len(CLASS_NAMES):
            name = CLASS_NAMES[label_idx]
        else:
            name = f'unknown_{label_idx}'

        cx, cy, cz = bboxes[i][:3]

        print(
            f'  {rank + 1:02d}: '
            f'label={label_idx}, '
            f'class={name}, '
            f'score={float(scores[i]):.3f}, '
            f'center=({cx:.2f}, {cy:.2f}, {cz:.2f})'
        )

    # =========================
    # 7. bbox 내부 point index 계산
    # =========================
    box_pts = get_box_points(coords_original, bboxes)

    # =========================
    # 8. 저장
    # =========================
    detections = dict(
        dataset_name=DATASET_NAME,
        points=pts_original,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        box_pts=box_pts,
        classes=CLASS_NAMES,
        split_mode='two_blocks',
        split_overlap=SPLIT_OVERLAP,
        superpoint_voxel_size=SUPERPOINT_VOXEL_SIZE,
    )

    with open(OUT, 'wb') as f:
        pickle.dump(detections, f)

    print(f'\n완료 → {OUT}')


if __name__ == '__main__':
    main()
