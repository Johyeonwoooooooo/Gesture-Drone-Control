# infer.py — .bin → UniDet3D detections .pkl, 모델 1회 로드 후 모든 REGION 일괄
import sys
#sys.path.insert(0, '/shareHost/minyoy/unidet3d')
sys.path.insert(0, '/home/jgshin22/work/Gesture-Drone-Control/unidet3d')  # unidet3d 경로로 수정

# mmdet3d보다 먼저 unidet3d 등록
from unidet3d.data_preprocessor import Det3DDataPreprocessor_
from unidet3d.unidet3d import UniDet3D

import numpy as np
import pickle
import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS

import scenes


CFG = '/home/jgshin22/work/Gesture-Drone-Control/unidet3d/configs/unidet3d_1xb8_scannet_s3dis_multiscan_3rscan_scannetpp_arkitscenes.py'
CKPT = '/home/jgshin22/work/Gesture-Drone-Control/unidet3d/work_dirs/unidet3d.pth'

# 여기만 바꾸면 다른 dataset head로도 실험 가능
# candidates: scannet, s3dis, multiscan, 3rscan, scannetpp, arkitscenes
DATASET_NAME = 'scannetpp'

SCORE_THR = 0.30
MAX_POINTS = 80000

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


def build_model():
    """모델 + class name을 1회 로드 (여러 region 에 재사용)."""
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
        class_names = SCANNETPP_CLASSES
    else:
        class_names = get_dataset_classes(cfg, DATASET_NAME)

    print(f'{DATASET_NAME} 클래스 수: {len(class_names)}')
    return model, class_names


def infer_region(region, model, class_names):
    """한 region .bin → detections .pkl."""
    from mmdet3d.structures import Det3DDataSample, PointData

    input_bin = scenes.bin_path(region)

    # 1. point cloud 로드
    pts_original = np.fromfile(input_bin, dtype=np.float32).reshape(-1, 9)
    coords_original = pts_original[:, :3]
    pts_original = pts_original[:, :6]  # model expects 6 channels (xyz + rgb)

    # 2. point 수 제한 (OOM 방지 — 이 단계가 webapp 백엔드에는 없음)
    pts, sampled_indices = random_sample_points(pts_original, max_points=MAX_POINTS)
    coords = pts[:, :3]

    # 3. voxel 기반 superpoint 생성
    sp_np = make_voxel_superpoints(coords, voxel_size=0.50)

    pts_tensor = torch.from_numpy(pts).float().cuda()
    sp = torch.from_numpy(sp_np).long().cuda()

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

    # 4. inference
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            results = model.predict(batch_inputs, [data_sample])

    result = results[0]

    bboxes = result.pred_instances_3d.bboxes_3d.tensor.detach().cpu().numpy()
    scores = result.pred_instances_3d.scores_3d.detach().cpu().numpy()
    labels = result.pred_instances_3d.labels_3d.detach().cpu().numpy()

    # 5. score threshold 적용
    mask = scores > SCORE_THR
    bboxes = bboxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    # 6. bbox 안에 들어가는 원본 point index 계산
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

    # 7. 저장
    detections = dict(
        dataset_name=DATASET_NAME,
        points=pts_original,
        sampled_points=pts,
        sampled_indices=sampled_indices,
        bboxes=bboxes,
        scores=scores,
        labels=labels,
        box_pts=box_pts,
        classes=class_names,
    )

    out = scenes.det_path(region)
    with open(out, 'wb') as f:
        pickle.dump(detections, f)
    print(f'[infer] {region}: {len(pts)} pts → {len(bboxes)} boxes → {out}')


def main():
    regions = sys.argv[1:] or scenes.REGIONS
    model, class_names = build_model()
    for i, region in enumerate(regions):
        print(f'=== ({i + 1}/{len(regions)}) {region} ===')
        infer_region(region, model, class_names)
    print('[infer] done.')


if __name__ == '__main__':
    main()
