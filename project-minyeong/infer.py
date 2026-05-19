# infer.py
import sys
sys.path.insert(0, '/data1/workspaces/jgshin22/Gesture-Drone-Control/unidet3d')

# mmdet3d보다 먼저 unidet3d 등록
from unidet3d.data_preprocessor import Det3DDataPreprocessor_
from unidet3d.unidet3d import UniDet3D

import numpy as np
import pickle
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS


CFG = '/data1/workspaces/jgshin22/Gesture-Drone-Control/unidet3d/configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py'
CKPT = '/data1/workspaces/jgshin22/Gesture-Drone-Control/unidet3d/work_dirs/unidet3d.pth'
INPUT = '/data1/workspaces/jgshin22/Gesture-Drone-Control/unidet3d/data/my_scene.bin'
OUT = '/data1/workspaces/jgshin22/Gesture-Drone-Control/unidet3d/data/detections.pkl'

# 여기만 바꾸면 다른 dataset head로도 실험 가능
# candidates: scannet, s3dis, multiscan, 3rscan, scannetpp, arkitscenes
DATASET_NAME = 'scannetpp'

SCANNETPP_CLASSES = [
    'table',
    'door',
    'ceiling lamp',
    'cabinet',
    'blinds',
    'curtain',
    'chair',
    'storage cabinet',
    'office chair',
    'bookshelf',
    'whiteboard',
    'window',
    'box',
    'monitor',
    'shelf',
    'heater',
    'kitchen cabinet',
    'sofa',
    'bed',
    'trash can',
    'book',
    'plant',
    'blanket',
    'tv',
    'computer tower',
    'refrigerator',
    'jacket',
    'sink',
    'bag',
    'picture',
    'pillow',
    'towel',
    'suitcase',
    'backpack',
    'crate',
    'keyboard',
    'rack',
    'toilet',
    'printer',
    'poster',
    'painting',
    'microwave',
    'shoes',
    'socket',
    'bottle',
    'bucket',
    'cushion',
    'basket',
    'shoe rack',
    'telephone',
    'file folder',
    'laptop',
    'plant pot',
    'exhaust fan',
    'cup',
    'coat hanger',
    'light switch',
    'speaker',
    'table lamp',
    'kettle',
    'smoke detector',
    'container',
    'power strip',
    'slippers',
    'paper bag',
    'mouse',
    'cutting board',
    'toilet paper',
    'paper towel',
    'pot',
    'clock',
    'pan',
    'tap',
    'jar',
    'soap dispenser',
    'binder',
    'bowl',
    'tissue box',
    'whiteboard eraser',
    'toilet brush',
    'spray bottle',
    'headphones',
    'stapler',
    'marker',
]

def make_voxel_superpoints(coords, voxel_size=0.10):
    """
    각 점을 개별 superpoint로 두지 않고,
    voxel grid 단위로 superpoint id를 부여한다.

    coords: (N, 3)
    return: sp_mask (N,)
    """
    xyz_min = coords.min(axis=0, keepdims=True)
    voxel_coord = np.floor((coords - xyz_min) / voxel_size).astype(np.int32)

    _, inverse = np.unique(voxel_coord, axis=0, return_inverse=True)
    return inverse.astype(np.int64)


def random_sample_points(pts, max_points=80000):
    """
    너무 큰 scene이면 point 수를 제한한다.
    원본 index도 같이 반환해서 나중에 필요할 때 복원 가능하게 한다.
    """
    n = len(pts)

    if n <= max_points:
        return pts, np.arange(n)

    choice = np.random.choice(n, max_points, replace=False)
    choice = np.sort(choice)

    return pts[choice], choice


def get_dataset_classes(cfg, dataset_name, model=None):
    """
    cfg에서 classes를 찾고, 없으면 fallback class name을 만든다.
    """
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

    dataset_order = ['scannet', 's3dis', 'multiscan', '3rscan', 'scannetpp', 'arkitscenes']

    if dataset_name in dataset_order:
        idx = dataset_order.index(dataset_name)
        if idx < len(datasets):
            metainfo = datasets[idx].get('metainfo', {})
            classes = metainfo.get('classes', None)
            if classes is not None and len(classes) > 0:
                return list(classes)

    print(f'[WARN] {dataset_name} classes를 config에서 찾지 못했습니다.')
    print('[WARN] 임시 class name을 자동 생성합니다.')

    # 일단 넉넉하게 200개 생성
    # labels가 실제로 몇 번까지 나오는지는 inference 후 확인 가능
    return [f'{dataset_name}_class_{i}' for i in range(200)]


def main():
    cfg = Config.fromfile(CFG)

    # mmengine registry에도 강제 등록
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

    if DATASET_NAME == 'scannetpp':
        CLASS_NAMES = SCANNETPP_CLASSES
    else:
        CLASS_NAMES = get_dataset_classes(cfg, DATASET_NAME)

    print(f'{DATASET_NAME} 클래스 수: {len(CLASS_NAMES)}')
    # =========================
    # 1. point cloud 로드
    # =========================
    _raw = np.fromfile(INPUT, dtype=np.float32)
    if _raw.size % 6 == 0:
        pts_original = _raw.reshape(-1, 6)
    elif _raw.size % 9 == 0:
        pts_original = _raw.reshape(-1, 9)[:, :6]
        print('[INFO] 9-channel bin detected, dropping normals (model expects 6 ch)')
    else:
        raise ValueError(f'cannot reshape {_raw.size} into (-1, 6) or (-1, 9)')
    coords_original = pts_original[:, :3]

    print(f'원본 point 수: {len(pts_original)}')

    # =========================
    # 2. point 수 제한
    # =========================
    pts, sampled_indices = random_sample_points(
        pts_original,
        max_points=80000,
    )
    coords = pts[:, :3]

    print(f'inference point 수: {len(pts)}')

    # =========================
    # 3. voxel 기반 superpoint 생성
    # =========================
    sp_np = make_voxel_superpoints(
        coords,
        voxel_size=0.10,
    )

    print(f'superpoint 수: {len(np.unique(sp_np))}')

    pts_tensor = torch.from_numpy(pts).float().cuda()
    sp = torch.from_numpy(sp_np).long().cuda()

    from mmdet3d.structures import Det3DDataSample, PointData

    data_sample = Det3DDataSample()

    # 중요:
    # 실제 INPUT 파일을 여기서 다시 읽는 게 아니라,
    # UniDet3D 내부 get_dataset()이 dataset 이름을 추론할 수 있도록
    # lidar_path 문자열 안에 scannetpp를 넣어준다.
    data_sample.set_metainfo({
        'lidar_path': f'{DATASET_NAME}/my_scene.bin',
        'pts_filename': f'{DATASET_NAME}/my_scene.bin',
        'file_name': f'{DATASET_NAME}/my_scene.bin',
        'box_type_3d': 'Depth',
        'box_mode_3d': 2,
    })

    gt_pts_seg = PointData()
    gt_pts_seg.sp_pts_mask = sp
    data_sample.gt_pts_seg = gt_pts_seg

    batch_inputs = dict(
        points=[pts_tensor],
        sp_pts_masks=[sp],
    )

    # =========================
    # 4. inference
    # =========================
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            results = model.predict(batch_inputs, [data_sample])

    result = results[0]

    bboxes = result.pred_instances_3d.bboxes_3d.tensor.detach().cpu().numpy()
    scores = result.pred_instances_3d.scores_3d.detach().cpu().numpy()
    labels = result.pred_instances_3d.labels_3d.detach().cpu().numpy()

    # =========================
    # 5. score threshold 적용
    # =========================
    score_thr = 0.30
    mask = scores > score_thr

    bboxes = bboxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    print(f'score threshold: {score_thr}')
    print(f'검출된 bbox: {len(bboxes)}개')

    # label preview
    print('상위 검출 결과:')
    for i in range(min(10, len(bboxes))):
        label_idx = int(labels[i])
        if 0 <= label_idx < len(CLASS_NAMES):
            class_name = CLASS_NAMES[label_idx]
        else:
            class_name = f'unknown_label_{label_idx}'

        print(
            f'  {i:02d}: '
            f'label={label_idx}, '
            f'class={class_name}, '
            f'score={float(scores[i]):.3f}'
        )

    # =========================
    # 6. bbox 안에 들어가는 원본 point index 계산
    # =========================
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

    # =========================
    # 7. 저장
    # =========================
    detections = dict(
        dataset_name=DATASET_NAME,
        points=pts_original,
        sampled_points=pts,
        sampled_indices=sampled_indices,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        box_pts=box_pts,
        classes=CLASS_NAMES,
    )

    with open(OUT, 'wb') as f:
        pickle.dump(detections, f)

    print(f'완료 → {OUT}')


if __name__ == '__main__':
    main()
