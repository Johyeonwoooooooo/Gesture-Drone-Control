"""
PointNet++ 3D Semantic Segmentation - 인터랙티브 뷰어
======================================================
Q → 다음 클래스 강조
A → 이전 클래스 강조
R → 전체 세그멘테이션 뷰 초기화

필요한 설치:
    pip install open3d "numpy<2.0"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
"""

import os
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
PLY_PATH        = "playground/3D/3d_map.ply"
NUM_POINTS      = 8192   # RTX 4060/4070 기준 최적값
NUM_CLASSES     = 13
PRETRAINED_PATH = "playground/3D/pointnet2_indoor.pth"

CLASS_NAMES = [
    'ceiling',    # 0
    'floor',      # 1
    'wall',       # 2
    'beam',       # 3
    'column',     # 4
    'window',     # 5
    'door',       # 6  ← 문!
    'table',      # 7
    'chair',      # 8
    'sofa',       # 9
    'bookcase',   # 10
    'board',      # 11
    'clutter',    # 12
]

CLASS_COLORS = np.array([
    [0.8, 0.8, 0.8],   # ceiling  - 밝은 회색
    [0.6, 0.4, 0.1],   # floor    - 갈색
    [0.7, 0.7, 0.5],   # wall     - 베이지
    [0.4, 0.4, 0.4],   # beam     - 어두운 회색
    [0.3, 0.3, 0.3],   # column   - 진회색
    [0.3, 0.7, 0.9],   # window   - 하늘색
    [1.0, 0.1, 0.1],   # door     - 빨간색
    [0.9, 0.8, 0.1],   # table    - 노란색
    [0.1, 0.7, 0.1],   # chair    - 초록색
    [0.8, 0.3, 0.8],   # sofa     - 보라색
    [0.3, 0.1, 0.9],   # bookcase - 진보라
    [0.9, 0.5, 0.1],   # board    - 주황색
    [0.5, 0.5, 0.5],   # clutter  - 중간 회색
], dtype=np.float64)

# 강조 안 된 포인트는 어둡게
DIM_COLOR = np.array([0.12, 0.12, 0.12], dtype=np.float64)


# ──────────────────────────────────────────────
# PointNet++ 모델
# ──────────────────────────────────────────────

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance  = torch.ones(B, N, device=device) * 1e10
    farthest  = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist     = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    vs = list(idx.shape); vs[1:] = [1] * (len(vs) - 1)
    rs = list(idx.shape); rs[0]  = 1
    bi = torch.arange(B, dtype=torch.long, device=device).view(vs).repeat(rs)
    return points[bi, idx, :]


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    d  = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    d +=  torch.sum(src ** 2, -1).view(B, N, 1)
    d +=  torch.sum(dst ** 2, -1).view(B, 1, M)
    return d


def ball_query(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    gi = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat([B, S, 1])
    gi[square_distance(new_xyz, xyz) > radius ** 2] = N
    gi = gi.sort(dim=-1)[0][:, :, :nsample]
    gf = gi[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    gi[gi == N] = gf[gi == N]
    return gi


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint, self.radius, self.nsample = npoint, radius, nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last = in_channel
        for out in mlp:
            self.mlp_convs.append(nn.Conv2d(last, out, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out))
            last = out

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        new_xyz = index_points(xyz, farthest_point_sample(xyz, self.npoint))
        idx     = ball_query(self.radius, self.nsample, xyz, new_xyz)
        grouped = index_points(xyz, idx) - new_xyz.view(B, self.npoint, 1, C)
        if points is not None:
            grouped = torch.cat([grouped, index_points(points, idx)], dim=-1)
        p = grouped.permute(0, 3, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            p = F.relu(bn(conv(p)))
        return new_xyz.permute(0, 2, 1), torch.max(p, 2)[0]


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last = in_channel
        for out in mlp:
            self.mlp_convs.append(nn.Conv1d(last, out, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out))
            last = out

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1    = xyz1.permute(0, 2, 1)
        xyz2    = xyz2.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        if S == 1:
            interp = points2.repeat(1, N, 1)
        else:
            dists, idx = square_distance(xyz1, xyz2).sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]
            w      = 1.0 / (dists + 1e-8)
            w      = w / w.sum(dim=2, keepdim=True)
            interp = (index_points(points2, idx) * w.unsqueeze(-1)).sum(dim=2)
        if points1 is not None:
            new_p = torch.cat([points1.permute(0, 2, 1), interp], dim=-1)
        else:
            new_p = interp
        new_p = new_p.permute(0, 2, 1)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_p = F.relu(bn(conv(new_p)))
        return new_p


class PointNet2SemSeg(nn.Module):
    def __init__(self, num_classes=13, in_channel=6):
        super().__init__()
        self.sa1  = PointNetSetAbstraction(1024, 0.1, 32, in_channel + 3, [32, 32, 64])
        self.sa2  = PointNetSetAbstraction(256,  0.2, 32, 64  + 3,        [64, 64, 128])
        self.sa3  = PointNetSetAbstraction(64,   0.4, 32, 128 + 3,        [128, 128, 256])
        self.sa4  = PointNetSetAbstraction(16,   0.8, 32, 256 + 3,        [256, 256, 512])
        self.fp4  = PointNetFeaturePropagation(768,  [256, 256])
        self.fp3  = PointNetFeaturePropagation(384,  [256, 256])
        self.fp2  = PointNetFeaturePropagation(320,  [256, 128])
        self.fp1  = PointNetFeaturePropagation(128,  [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1   = nn.BatchNorm1d(128)
        self.drop  = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0p = xyz;  l0x = xyz[:, :3, :]
        l1x, l1p = self.sa1(l0x, l0p)
        l2x, l2p = self.sa2(l1x, l1p)
        l3x, l3p = self.sa3(l2x, l2p)
        l4x, l4p = self.sa4(l3x, l3p)
        l3p = self.fp4(l3x, l4x, l3p, l4p)
        l2p = self.fp3(l2x, l3x, l2p, l3p)
        l1p = self.fp2(l1x, l2x, l1p, l2p)
        l0p = self.fp1(l0x, l1x, None, l1p)
        x   = self.drop(F.relu(self.bn1(self.conv1(l0p))))
        return F.log_softmax(self.conv2(x), dim=1)


# ──────────────────────────────────────────────
# 데이터 로드 & 추론
# ──────────────────────────────────────────────

def load_ply(ply_path):
    print(f"[1/3] PLY 로드: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        raise ValueError("포인트가 없습니다. 경로를 확인하세요.")
    print(f"      총 포인트: {len(pts):,}개")

    if pcd.has_colors():
        cols = np.asarray(pcd.colors).copy()
        print("      색상: RGB 있음")
    else:
        z   = pts[:, 2]
        z_n = (z - z.min()) / (z.max() - z.min() + 1e-8)
        cols = np.column_stack([z_n, z_n, z_n])
        print("      색상: 없음 → 높이값으로 대체")

    # 정규화 (전체 포인트 기준)
    centroid = pts.mean(axis=0)
    pts_norm = (pts - centroid) / (np.max(np.linalg.norm(pts - centroid, axis=1)) + 1e-8)

    return pcd, pts_norm, cols


def run_inference_full(model, pts_norm, cols, device, chunk_size=NUM_POINTS):
    """
    전체 포인트를 chunk_size개씩 랜덤 샘플링해서 반복 추론.
    각 포인트마다 클래스별 득표 누적 후 최다득표 클래스로 결정 (voting).
    """
    N          = len(pts_norm)
    num_chunks = max(1, round(N / chunk_size))
    vote_counts = np.zeros((N, NUM_CLASSES), dtype=np.int32)

    print(f"[2/3] 전체 추론 중... ({num_chunks}번 청크, 청크당 {chunk_size:,}개)")
    model.eval()

    with torch.no_grad():
        for i in range(num_chunks):
            idx   = np.random.choice(N, chunk_size, replace=(N < chunk_size))
            pts_s = pts_norm[idx]
            col_s = cols[idx]

            normals = np.zeros_like(pts_s)  # normal 없으면 zeros 패딩
            data   = np.concatenate([pts_s, col_s, normals], axis=1).T  # (9, N)
            tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
            pred   = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()

            for j, label in zip(idx, pred):
                vote_counts[j, label] += 1

            if (i + 1) % 10 == 0 or (i + 1) == num_chunks:
                covered = (vote_counts.sum(axis=1) > 0).sum()
                print(f"      {i+1:3d}/{num_chunks}  커버: {covered:,}/{N:,} ({covered/N*100:.1f}%)")

    # 한 번도 뽑히지 않은 포인트 → 최근접 이웃 레이블로 채움
    unlabeled = vote_counts.sum(axis=1) == 0
    if unlabeled.sum() > 0:
        print(f"      미커버 {unlabeled.sum():,}개 → KNN 보간")
        from sklearn.neighbors import KDTree
        labeled_idx   = np.where(~unlabeled)[0]
        unlabeled_idx = np.where(unlabeled)[0]
        tree = KDTree(pts_norm[labeled_idx])
        _, nn = tree.query(pts_norm[unlabeled_idx], k=1)
        vote_counts[unlabeled_idx] = vote_counts[labeled_idx[nn.flatten()]]

    print("      추론 완료!")
    return vote_counts.argmax(axis=1)   # (N,)


# ──────────────────────────────────────────────
# 인터랙티브 뷰어
# ──────────────────────────────────────────────

class InteractiveViewer:
    """
    원본 PLY를 그대로 보여주다가,
    Q/A로 클래스 선택 시 해당 포인트만 클래스 색상으로 덮어씌움.

    mode == -1  : 원본 PLY 색상 그대로
    mode == 0~12: 해당 클래스 포인트만 클래스 색상, 나머지는 원본 RGB 유지
    """

    def __init__(self, full_pcd, pred_labels, base_colors):
        """
        full_pcd    : 원본 PLY o3d.geometry.PointCloud
        pred_labels : 전체 포인트별 예측 레이블 (M,)
        base_colors : 원본 RGB (M, 3)
        """
        self.full_pcd    = full_pcd
        self.pred_labels = pred_labels
        self.base_colors = base_colors
        self.mode        = -1

        self.counts = [(pred_labels == i).sum() for i in range(NUM_CLASSES)]
        self._print_summary()

    def _print_summary(self):
        print("\n[세그멘테이션 결과]")
        print("─" * 40)
        total = len(self.pred_labels)
        for i, name in enumerate(CLASS_NAMES):
            cnt = self.counts[i]
            if cnt == 0:
                continue
            pct = cnt / total * 100
            tag = "  ★ 문!" if i == 6 else ""
            print(f"  [{i:2d}] {name:12s}: {cnt:6,}개  ({pct:5.1f}%){tag}")
        print("─" * 40)
        print("\n[ 조작 방법 ]")
        print("  Q   → 다음 클래스 강조")
        print("  A   → 이전 클래스 강조")
        print("  R   → 원본 뷰로 초기화")
        print("  ESC → 종료\n")

    def _apply_colors(self):
        """full_pcd의 색상을 현재 모드에 맞게 업데이트"""
        if self.mode == -1:
            colors = self.base_colors.copy()
        else:
            colors = self.base_colors * 0.15
            colors[self.pred_labels == self.mode] = CLASS_COLORS[self.mode]

        self.full_pcd.colors = o3d.utility.Vector3dVector(colors)

    def _print_current(self):
        if self.mode == -1:
            print("▶  원본 PLY 뷰")
        else:
            cnt = self.counts[self.mode]
            pct = cnt / len(self.pred_labels) * 100
            print(f"▶  [{self.mode:2d}] {CLASS_NAMES[self.mode]}  —  {cnt:,}개 ({pct:.1f}%)")

    def run(self):
        print("[3/3] 뷰어 시작\n")

        # 초기 색상 적용 (원본 그대로)
        self._apply_colors()

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(
            window_name="PointNet++ Segmentation  |  Q: 다음  A: 이전  R: 원본  ESC: 종료",
            width=1400, height=800
        )
        vis.add_geometry(self.full_pcd)

        opt = vis.get_render_option()
        opt.point_size       = 1.5
        opt.background_color = np.array([0.05, 0.05, 0.05])

        def refresh(vis):
            self._apply_colors()
            vis.update_geometry(self.full_pcd)
            vis.poll_events()
            vis.update_renderer()
            self._print_current()

        def key_q(vis):
            self.mode = 0 if self.mode == -1 else (self.mode + 1) % NUM_CLASSES
            refresh(vis)

        def key_a(vis):
            self.mode = NUM_CLASSES - 1 if self.mode == -1 else (self.mode - 1) % NUM_CLASSES
            refresh(vis)

        def key_r(vis):
            self.mode = -1
            refresh(vis)

        vis.register_key_callback(ord("Q"), key_q)
        vis.register_key_callback(ord("A"), key_a)
        vis.register_key_callback(ord("R"), key_r)

        self._print_current()
        vis.run()
        vis.destroy_window()


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}")
    print("=" * 40)

    # 전체 PLY 로드
    full_pcd, pts_norm, base_colors = load_ply(PLY_PATH)

    # 모델
    model = PointNet2SemSeg(num_classes=NUM_CLASSES, in_channel=9).to(device)  # xyz+rgb+normal(zeros)
    if os.path.exists(PRETRAINED_PATH):
        ckpt = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
        # 체크포인트 형식에 따라 자동 처리
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"  가중치 로드: {PRETRAINED_PATH}")
            print(f"  epoch: {ckpt.get('epoch', '?')}, IoU: {ckpt.get('class_avg_iou', '?'):.4f}")
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
            print(f"  가중치 로드: {PRETRAINED_PATH}")
        else:
            model.load_state_dict(ckpt)
            print(f"  가중치 로드: {PRETRAINED_PATH}")
    else:
        print("  [주의] 가중치 없음 → 랜덤 초기화 (파이프라인 테스트용)")
        print("  pretrained: https://github.com/yanx27/Pointnet_Pointnet2_pytorch\n")

    # 전체 포인트 청크 추론
    pred_labels = run_inference_full(model, pts_norm, base_colors, device, chunk_size=NUM_POINTS)

    # 뷰어 실행
    viewer = InteractiveViewer(full_pcd, pred_labels, base_colors)
    viewer.run()


if __name__ == "__main__":
    main()