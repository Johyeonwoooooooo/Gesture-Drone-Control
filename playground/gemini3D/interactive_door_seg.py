"""
Gemini 3D Interactive Door Segmentation
========================================
사용법:
    - R: 원본 PLY 색상 보기
    - D: '문(Door)' 클래스만 빨간색으로 강조 (나머지는 어둡게)
    - Shift + 마우스 클릭: 클릭한 포인트의 클래스를 강조
    - Q/A: 클래스 순환 강조
    - ESC: 종료
"""

import os
import sys
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F

# 상위 폴더의 파일을 참조하기 위해 경로 설정 (필요 시)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 설정
PLY_PATH        = "../../playground/3D/3d_map.ply"
PRETRAINED_PATH = "../../playground/3D/pointnet2_indoor.pth"
CACHE_PATH      = "seg_labels_cache.npy" # 추론 결과 저장 파일
NUM_POINTS      = 16384 # 8192 -> 16384로 상향 (RTX 4060 Ti 최적화)
NUM_CLASSES     = 13

CLASS_NAMES = [
    'ceiling', 'floor', 'wall', 'beam', 'column', 'window', 
    'door', # index 6
    'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'
]

CLASS_COLORS = np.array([
    [0.8, 0.8, 0.8], [0.6, 0.4, 0.1], [0.7, 0.7, 0.5], [0.4, 0.4, 0.4], 
    [0.3, 0.3, 0.3], [0.3, 0.7, 0.9], [1.0, 0.0, 0.0], [0.9, 0.8, 0.1], 
    [0.1, 0.7, 0.1], [0.8, 0.3, 0.8], [0.3, 0.1, 0.9], [0.9, 0.5, 0.1], 
    [0.5, 0.5, 0.5]
], dtype=np.float64)

# ------------------------------------------------------------------------------
# PointNet++ 모델 정의 (self-contained)
# ------------------------------------------------------------------------------

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape); view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape); repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def ball_query(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    sq_dist = square_distance(new_xyz, xyz)
    group_idx[sq_dist > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    group_idx[group_idx == N] = group_first[group_idx == N]
    return group_idx

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) - new_xyz.view(B, self.npoint, 1, C)
        if points is not None:
            grouped_points = index_points(points, idx)
            grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        grouped_points = grouped_points.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_points = F.relu(bn(conv(grouped_points)))
        new_points = torch.max(grouped_points, 2)[0]
        return new_xyz.permute(0, 2, 1), new_points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1, xyz2, points2 = xyz1.permute(0, 2, 1), xyz2.permute(0, 2, 1), points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists, idx = square_distance(xyz1, xyz2).sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            weight = 1.0 / (dists + 1e-8)
            weight = weight / torch.sum(weight, dim=2, keepdim=True)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.unsqueeze(-1), dim=2)
        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = F.relu(bn(conv(new_points)))
        return new_points

class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes=13, in_channel=9):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, in_channel + 3, [32, 32, 64])
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128])
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256])
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512])
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0p = xyz
        l0x = xyz[:, :3, :]
        l1x, l1p = self.sa1(l0x, l0p)
        l2x, l2p = self.sa2(l1x, l1p)
        l3x, l3p = self.sa3(l2x, l2p)
        l4x, l4p = self.sa4(l3x, l3p)
        l3p = self.fp4(l3x, l4x, l3p, l4p)
        l2p = self.fp3(l2x, l3x, l2p, l3p)
        l1p = self.fp2(l1x, l2x, l1p, l2p)
        l0p = self.fp1(l0x, l1x, None, l1p)
        x = self.drop(F.relu(self.bn1(self.conv1(l0p))))
        return F.log_softmax(self.conv2(x), dim=1)

# ------------------------------------------------------------------------------
# 데이터 로드 & 추론 유틸리티
# ------------------------------------------------------------------------------

def load_ply_and_norm(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
    else:
        cols = np.ones_like(pts) * 0.5
    centroid = pts.mean(axis=0)
    pts_norm = (pts - centroid) / (np.max(np.linalg.norm(pts - centroid, axis=1)) + 1e-8)
    return pcd, pts_norm, cols

def run_inference(model, pts_norm, cols, device):
    N = len(pts_norm)
    vote_counts = np.zeros((N, NUM_CLASSES), dtype=np.int32)
    model.eval()
    # 기존 * 2 에서 * 0.8로 하향 (속도 약 2.5배 향상)
    num_chunks = max(1, round(N / NUM_POINTS * 0.8))
    print(f"추론 시작... ({num_chunks}회 샘플링)")
    with torch.no_grad():
        for i in range(num_chunks):
            idx = np.random.choice(N, NUM_POINTS, replace=(N < NUM_POINTS))
            pts_s = pts_norm[idx]
            col_s = cols[idx]
            data = np.concatenate([pts_s, col_s, np.zeros_like(pts_s)], axis=1).T
            tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            pred = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
            for j, label in zip(idx, pred):
                vote_counts[j, label] += 1
    
    unlabeled = vote_counts.sum(axis=1) == 0
    if unlabeled.sum() > 0:
        from sklearn.neighbors import KDTree
        tree = KDTree(pts_norm[~unlabeled])
        _, nn = tree.query(pts_norm[unlabeled], k=1)
        vote_counts[unlabeled] = vote_counts[np.where(~unlabeled)[0][nn.flatten()]]
        
    return vote_counts.argmax(axis=1)

# ------------------------------------------------------------------------------
# 인터랙티브 뷰어
# ------------------------------------------------------------------------------

class InteractiveDoorSeg:
    def __init__(self, pcd, labels, base_colors):
        self.pcd = pcd
        self.labels = labels
        self.base_colors = base_colors
        self.mode = -1 # -1: Original, 0~12: Class
        
        print("\n" + "="*50)
        print(" Gemini 3D Door Segmentation Viewer")
        print("="*50)
        print(" [D] : 문(Door) 강조 (빨간색)")
        print(" [R] : 원본 색상 복구")
        print(" [Shift + Click] : 포인트 선택 시 해당 클래스 강조")
        print(" [Q/A] : 클래스 순환 강조")
        print(" [ESC] : 종료")
        print("="*50)

    def update_colors(self):
        if self.mode == -1:
            colors = self.base_colors
        else:
            colors = self.base_colors * 0.1 # Background dimming
            mask = self.labels == self.mode
            colors[mask] = CLASS_COLORS[self.mode]
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="Gemini 3D Door Control", width=1280, height=720)
        vis.add_geometry(self.pcd)
        
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])

        def refresh(vis):
            self.update_colors()
            vis.update_geometry(self.pcd)
            print(f"현재 모드: {'원본' if self.mode == -1 else CLASS_NAMES[self.mode]}")
            return False

        # 콜백 함수 정의 (튜플 반환 오류 방지)
        def key_d(vis):
            self.mode = 6
            return refresh(vis)

        def key_r(vis):
            self.mode = -1
            return refresh(vis)

        def key_q(vis):
            self.mode = 0 if self.mode == -1 else (self.mode + 1) % NUM_CLASSES
            return refresh(vis)

        def key_a(vis):
            self.mode = NUM_CLASSES - 1 if self.mode == -1 else (self.mode - 1) % NUM_CLASSES
            return refresh(vis)

        # 키 콜백 등록
        vis.register_key_callback(ord("D"), key_d)
        vis.register_key_callback(ord("R"), key_r)
        vis.register_key_callback(ord("Q"), key_q)
        vis.register_key_callback(ord("A"), key_a)

        vis.run()
        vis.destroy_window()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")

    # 파일 경로 체크
    abs_ply = os.path.abspath(os.path.join(os.path.dirname(__file__), PLY_PATH))
    abs_model = os.path.abspath(os.path.join(os.path.dirname(__file__), PRETRAINED_PATH))
    abs_cache = os.path.abspath(os.path.join(os.path.dirname(__file__), CACHE_PATH))

    if not os.path.exists(abs_ply):
        print(f"Error: {abs_ply} 파일을 찾을 수 없습니다.")
        return

    pcd, pts_norm, base_colors = load_ply_and_norm(abs_ply)
    
    # 캐시 확인
    if os.path.exists(abs_cache):
        print(f"\n[알림] 저장된 추론 결과 발견: {abs_cache}")
        print("결과를 불러오는 중... (추론 단계 건너뜀)")
        labels = np.load(abs_cache)
        if len(labels) != len(pts_norm):
            print("Warning: 캐시 데이터와 PLY 포인트 수가 다릅니다. 다시 추론합니다.")
            labels = None
    else:
        labels = None

    if labels is None:
        model = PointNet2SemSeg(num_classes=NUM_CLASSES, in_channel=9).to(device)
        if os.path.exists(abs_model):
            print(f"가중치 로드 중: {abs_model}")
            ckpt = torch.load(abs_model, map_location=device, weights_only=False)
            if isinstance(ckpt, dict):
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                elif 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'])
                else:
                    model.load_state_dict(ckpt)
            else:
                model.load_state_dict(ckpt)
        else:
            print("Warning: 가중치 파일을 찾을 수 없어 랜덤 초기값으로 실행합니다.")

        labels = run_inference(model, pts_norm, base_colors, device)
        
        # 결과 저장
        print(f"추론 결과 저장 중: {abs_cache}")
        np.save(abs_cache, labels)
    
    viewer = InteractiveDoorSeg(pcd, labels, base_colors)
    viewer.run()

if __name__ == "__main__":
    main()
