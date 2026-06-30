"""
실내 공간 스캔 npy 데이터 → LitePT 시맨틱 세그멘테이션 → 물체별 중심점 추출

입력: coord.npy, color.npy  (data/npy/<room>/ 형식)
출력: detections.pkl  { class_name, center, point_indices, ... }

사용법:
    python infer_centers.py --npy_dir project/data/npy/ROOM_NAME
    python infer_centers.py --npy_dir project/data/npy/ROOM_NAME --out result.pkl
    python infer_centers.py --npy_root project/data/npy  # 전체 방 일괄 처리
"""
import sys
import os
import argparse
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# ── LitePT 경로 등록 ──────────────────────────────────────
LITEPT_ROOT = Path(__file__).parent / 'LitePT'
sys.path.insert(0, str(LITEPT_ROOT))

from huggingface_hub import hf_hub_download

# ============================================================
# 설정
# ============================================================
HF_REPO_ID   = 'prs-eth/LitePT'
HF_CKPT_PATH = 'scannet-semseg-litept-small-v1m1/model/model_best.pth'

GRID_SIZE   = 0.02   # GridSample voxel (학습 설정과 동일)
MAX_POINTS  = 10_000_000  # 실질적 제한 없음 — 4090 24GB 여유

# DBSCAN 설정 (중심점 클러스터링)
DBSCAN_EPS        = 0.2   # m 단위
DBSCAN_MIN_POINTS = 200    # 최소 포인트 수

# 무시할 배경 클래스 (wall, floor)
IGNORE_CLASSES = {'wall', 'floor'}

SCANNET_CLASSES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
    'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
    'bathtub', 'otherfurniture',
]


# ============================================================
# 모델 구성 — spconv 없이 LitePT 백본 + Linear 헤드 직접 구성
# ============================================================
class LitePTSegmentor(torch.nn.Module):
    """DefaultSegmentorV2와 동일한 구조를 spconv 없이 재현."""
    def __init__(self, backbone, num_classes, backbone_out_channels):
        super().__init__()
        self.backbone = backbone
        self.seg_head = torch.nn.Linear(backbone_out_channels, num_classes)

    def forward(self, input_dict):
        from litept.model import Point
        point = Point(input_dict)
        point = self.backbone(point)
        # decoder 거슬러 올라가기 (GridPooling 역방향)
        while 'pooling_parent' in point.keys():
            parent  = point.pop('pooling_parent')
            inverse = point.pop('pooling_inverse')
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        return self.seg_head(point.feat)   # (N, num_classes)


def build_model():
    from litept.model import LitePT

    backbone = LitePT(
        in_channels=6,
        order=('z', 'z-trans', 'hilbert', 'hilbert-trans'),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    )

    return LitePTSegmentor(
        backbone=backbone,
        num_classes=len(SCANNET_CLASSES),
        backbone_out_channels=72,
    )


# ============================================================
# 데이터 전처리
# ============================================================
def load_and_preprocess(npy_dir: Path):
    """npy → LitePT 입력 dict 변환."""
    coord  = np.load(npy_dir / 'coord.npy').astype(np.float32)
    color  = np.load(npy_dir / 'color.npy').astype(np.float32)
    normal = np.load(npy_dir / 'normal.npy').astype(np.float32)

    # NormalizeColor: [0,255] → [0,1]
    if color.max() > 1.0:
        color = color / 255.0

    # 원본 좌표 보존 (중심점·시각화용)
    coord_orig_world = coord.copy()

    # CenterShift(apply_z=True): xyz 중심 이동 (모델 입력용)
    coord = coord - coord.mean(axis=0)

    n_orig = len(coord)

    # ── GridSample (voxel 다운샘플) ───────────────────────
    voxel_idx = np.floor(coord / GRID_SIZE).astype(np.int64)
    _, unique_idx, inverse_idx = np.unique(
        voxel_idx, axis=0, return_index=True, return_inverse=True
    )
    coord_down  = coord[unique_idx]
    color_down  = color[unique_idx]
    normal_down = normal[unique_idx]
    grid_coord  = np.floor(coord_down / GRID_SIZE).astype(np.int64)
    # spconv은 0-origin 양수 좌표를 요구함
    grid_coord  = grid_coord - grid_coord.min(axis=0)

    n_down = len(coord_down)
    print(f'  GridSample: {n_orig} → {n_down} points')

    # ── MaxPoints 제한 ────────────────────────────────────
    if n_down > MAX_POINTS:
        choice = np.random.choice(n_down, MAX_POINTS, replace=False)
        choice = np.sort(choice)
        coord_down  = coord_down[choice]
        color_down  = color_down[choice]
        normal_down = normal_down[choice]
        grid_coord  = grid_coord[choice]
        remap = np.full(n_down, -1, dtype=np.int64)
        remap[choice] = np.arange(len(choice))
        inverse_idx = remap[inverse_idx]
        n_down = len(coord_down)
        print(f'  MaxPoints: → {n_down} points')

    # feat = color + normal (학습 시 feat_keys=("color","normal"))
    feat = np.concatenate([color_down, normal_down], axis=1).astype(np.float32)

    data = dict(
        coord=torch.from_numpy(coord_down).float(),
        grid_coord=torch.from_numpy(grid_coord).long(),
        feat=torch.from_numpy(feat).float(),
        offset=torch.tensor([n_down], dtype=torch.long),
    )

    return data, coord_orig_world, inverse_idx


# ============================================================
# 중심점 추출 (DBSCAN 클러스터링)
# ============================================================
def extract_centers(coord_orig, pred_labels_orig, ignore=IGNORE_CLASSES):
    """
    per-point label → 클래스별 DBSCAN → 인스턴스 중심점 반환.

    반환:
        list of dict {
            class_name, label_idx, center (3,),
            point_indices, n_points
        }
    """
    from sklearn.cluster import DBSCAN

    results = []

    for label_idx, class_name in enumerate(SCANNET_CLASSES):
        if class_name in ignore:
            continue

        mask = pred_labels_orig == label_idx
        if mask.sum() < DBSCAN_MIN_POINTS:
            continue

        pts = coord_orig[mask]
        pt_indices = np.where(mask)[0]

        db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_POINTS, n_jobs=-1)
        cluster_ids = db.fit_predict(pts)

        for cid in np.unique(cluster_ids):
            if cid == -1:   # noise
                continue

            c_mask = cluster_ids == cid
            c_pts  = pts[c_mask]
            c_idx  = pt_indices[c_mask]

            center = c_pts.mean(axis=0)

            results.append(dict(
                class_name=class_name,
                label_idx=label_idx,
                center=center,
                point_indices=c_idx,
                n_points=int(c_mask.sum()),
            ))

    results.sort(key=lambda x: x['n_points'], reverse=True)
    return results


# ============================================================
# 방 하나 처리
# ============================================================
def process_room(model, npy_dir: Path, out_path: Path, args_skip_existing: bool = False):
    print(f'\n{"="*60}')
    print(f'[ROOM] {npy_dir.name}')

    if out_path.exists() and args_skip_existing:
        print(f'  [SKIP] {out_path} 이미 존재')
        return

    # ── 전처리 ────────────────────────────────────────────
    data, coord_orig, inverse_idx = load_and_preprocess(npy_dir)

    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.cuda()

    # ── 추론 ──────────────────────────────────────────────
    with torch.no_grad():
        logits = model(data)  # (N_down, num_classes)

    pred_down = logits.argmax(dim=1).cpu().numpy()  # (N_down,)

    # 원본 포인트로 복원
    valid = inverse_idx >= 0
    pred_orig = np.full(len(coord_orig), -1, dtype=np.int64)
    pred_orig[valid] = pred_down[inverse_idx[valid]]

    # ── 중심점 추출 ───────────────────────────────────────
    centers = extract_centers(coord_orig, pred_orig)

    print(f'  검출된 인스턴스: {len(centers)}개')
    for i, obj in enumerate(centers[:15]):
        cx, cy, cz = obj['center']
        print(
            f'  {i+1:02d}: {obj["class_name"]:20s} '
            f'n={obj["n_points"]:5d}  '
            f'center=({cx:.2f}, {cy:.2f}, {cz:.2f})'
        )

    # ── 저장 ──────────────────────────────────────────────
    result = dict(
        room=npy_dir.name,
        classes=SCANNET_CLASSES,
        pred_labels=pred_orig,       # (N_orig,) per-point label
        centers=centers,             # list of instance dicts
        coord=coord_orig,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    print(f'  저장 → {out_path}')


# ============================================================
# 모델 로드
# ============================================================
def load_model():
    print('모델 구성 중...')
    model = build_model()

    print(f'사전학습 가중치 다운로드: {HF_CKPT_PATH}')
    ckpt_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CKPT_PATH,
        repo_type='model',
    )

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt.get('state_dict', ckpt)

    # checkpoint 키 prefix 제거 (module. or module.backbone. 등)
    clean = OrderedDict()
    for k, v in state.items():
        # "module." prefix 제거
        new_k = k.lstrip('module.')
        clean[new_k] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f'  [WARN] missing keys ({len(missing)}): {missing[:5]}...')
    if unexpected:
        print(f'  [WARN] unexpected keys ({len(unexpected)}): {unexpected[:5]}...')

    model = model.cuda()
    model.eval()
    print('모델 준비 완료')
    return model


# ============================================================
# 메인
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--npy_dir',  type=Path, help='방 하나 (coord.npy가 있는 폴더)')
    group.add_argument('--npy_root', type=Path, help='전체 방 일괄 처리 (하위 폴더 순회)')
    parser.add_argument('--out', type=Path, default=None,
                        help='출력 pkl 경로 (--npy_dir 단독 사용 시)')
    parser.add_argument('--skip_existing', action='store_true', default=False)
    args = parser.parse_args()

    model = load_model()

    if args.npy_dir:
        out = args.out or (args.npy_dir / 'centers.pkl')
        process_room(model, args.npy_dir, out, args.skip_existing)

    else:
        rooms = sorted([
            p for p in args.npy_root.iterdir()
            if p.is_dir() and (p / 'coord.npy').exists()
        ])
        print(f'총 {len(rooms)}개 방 처리 시작')

        for room in rooms:
            try:
                process_room(model, room, room / 'centers.pkl', args.skip_existing)
            except Exception as e:
                print(f'  [ERROR] {room.name}: {e}')

        print('\n모든 방 처리 완료')


if __name__ == '__main__':
    main()
