import os
import sys
import numpy as np
import torch
import importlib
import time
import clip

BASE_DIR = '/shareHost/minyoy/votenet'
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'scannet'))

from plyfile import PlyData
from pc_util import random_sampling
from ap_helper import parse_predictions
from scannet_detection_dataset import DC

# ── 설정 ──────────────────────────────────────────
PLY_PATH        = os.path.join(BASE_DIR, 'demo_files/input_pc_hm3d.ply')
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'demo_files/pretrained_votenet_on_scannet.tar')
NUM_POINT       = 40000
CONF_THRESH     = 0.5

QUERIES = [
    'find the bed',
    'where is the chair',
    'find the door',
    'find the window',
    'find the table',
]
# ──────────────────────────────────────────────────

SCANNET_CLASSES = [DC.class2type[i] for i in range(DC.num_class)]


def read_ply_xyz(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    return np.stack([np.array(pc['x']), np.array(pc['y']), np.array(pc['z'])], axis=1)


def preprocess_point_cloud(point_cloud, num_point=40000):
    # 변환 없이 원본 HM3D 좌표 그대로 사용
    # height feature만 추가 (VoteNet이 주로 사용하는 feature)
    point_cloud = point_cloud[:, 0:3]
    floor_height = np.percentile(point_cloud[:, 2], 0.99)
    height = point_cloud[:, 2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
    point_cloud = random_sampling(point_cloud, num_point)
    return np.expand_dims(point_cloud.astype(np.float32), 0)


def text_to_class(query, clip_model):
    query_input  = clip.tokenize([query]).cuda()
    class_inputs = clip.tokenize(SCANNET_CLASSES).cuda()
    with torch.no_grad():
        query_feat  = clip_model.encode_text(query_input)
        class_feats = clip_model.encode_text(class_inputs)
    query_feat  = query_feat  / query_feat.norm(dim=-1, keepdim=True)
    class_feats = class_feats / class_feats.norm(dim=-1, keepdim=True)
    similarities = (query_feat @ class_feats.T).squeeze(0)
    best_idx = similarities.argmax().item()
    return best_idx, SCANNET_CLASSES[best_idx], similarities[best_idx].item()


def load_votenet(checkpoint_path, device):
    MODEL = importlib.import_module('votenet')
    net = MODEL.VoteNet(
        num_proposal=256, input_feature_dim=1, vote_factor=1,
        sampling='seed_fps', num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    return net


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('모델 로드 중...')
    net = load_votenet(CHECKPOINT_PATH, device)
    clip_model, _ = clip.load('ViT-B/32', device=device)
    print('모델 로드 완료!\n')

    print(f'포인트 클라우드 로드: {PLY_PATH}')
    point_cloud = read_ply_xyz(PLY_PATH)
    pc = preprocess_point_cloud(point_cloud, NUM_POINT)
    pc_tensor = torch.from_numpy(pc).to(device)

    print('VoteNet 추론 중...')
    tic = time.time()
    with torch.no_grad():
        end_points = net({'point_clouds': pc_tensor})
    toc = time.time()
    end_points['point_clouds'] = pc_tensor
    print(f'추론 시간: {toc - tic:.3f}초')

    eval_config_dict = {
        'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
        'use_old_type_nms': False, 'cls_nms': False, 'per_class_proposal': False,
        'conf_thresh': CONF_THRESH, 'dataset_config': DC,
    }
    pred_map_cls = parse_predictions(end_points, eval_config_dict)
    detections = pred_map_cls[0]
    print(f'총 감지된 물체: {len(detections)}개\n')

    print('감지된 전체 목록:')
    for cls_id, bbox, score in detections:
        center = bbox.mean(axis=0) if bbox.ndim > 1 else bbox[:3]
        print(f'  {DC.class2type[cls_id]:15s} score={score:.2f}  '
              f'center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})')

    print('\n' + '=' * 55)
    print('텍스트 쿼리 결과')
    print('=' * 55)

    for query in QUERIES:
        cls_idx, cls_name, clip_score = text_to_class(query, clip_model)
        print(f'\n쿼리: "{query}"')
        print(f'  CLIP 매칭: {cls_name} (similarity={clip_score:.3f})')

        matched = [(bbox, score) for cid, bbox, score in detections if cid == cls_idx]

        if matched:
            best_bbox, best_score = max(matched, key=lambda x: x[1])
            center = best_bbox.mean(axis=0) if best_bbox.ndim > 1 else best_bbox[:3]
            print(f'  감지 개수: {len(matched)}개')
            print(f'  최고 score: {best_score:.3f}')
            print(f'  3D 중심 좌표: x={center[0]:.3f}, y={center[1]:.3f}, z={center[2]:.3f}')
        else:
            print(f'  감지되지 않음')


if __name__ == '__main__':
    main()
