import os
import argparse
import copy
from dataclasses import dataclass
from bisect import bisect_left
from collections import deque
from typing import List, Tuple, Optional, Dict

import numpy as np # 모든 3D 변환
import open3d as o3d # SLAM 핵심 라이브러리
from scipy.spatial.transform import Rotation # IMU orientation 처리
from PIL import Image # PNG 파일 로딩
import skvideo.io # 영상 기반 실험을 위한 프레임 읽기용

# --- Configuration(상수 정의 및 numpy 호환성 패치) ---
# StrayScanner depth 해상도 (256*192)
DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0 # IR 기반 LiDAR 최대 측정 거리를 반영

# numpy 1.24 이상에서 np.float, np.int가 제거되어서
# YOLO/Open3D 동일성 유지 위해 패치
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

# IMU 상태 벡터(state)를 정의하기 위한 dataclass
@dataclass
class ImuState:
    R: np.ndarray   # 3x3 rotation matrix (IMU가 바라보는 방향)
    v: np.ndarray   # 3D velocity (회전이 보정된 IMU 속도)
    p: np.ndarray   # 3D position (IMU의 월드 좌표 위치)

# --- Global Map (Keyframe Strategy) ---
# Keyframe 기반의 global map 생성이라는 구조를 가짐
class GlobalMap:
    def __init__(self, voxel_size=0.03):    # voxel_size=0.03: 맵이 너무 조밀해지지 않도록 3cm 단위로 다운샘플
        self.voxel_size = voxel_size
        self.global_pcd = o3d.geometry.PointCloud() # SLAM 누적 포인트클라우드를 저장하는 최종 월드 맵
        self.last_keyframe_pose = np.eye(4) # 최근 키프레임의 pose(4*4 matrix)
        # [튜닝] 맵이 과도하게 커지는 것을 방지하기 위해 키프레임 간격을 조금 넓힘
        self.keyframe_dist_thresh = 0.1  # 10cm 이동 시 맵 추가
        self.keyframe_angle_thresh = 0.1 # 약 5도 회전 시 맵 추가           

    # 키프레임을 추가할지 결정
    def needs_update(self, curr_pose):
        # transition difference
        delta_trans = np.linalg.norm(curr_pose[:3, 3] - self.last_keyframe_pose[:3, 3])
        
        # rotation difference
        R_diff = np.linalg.inv(self.last_keyframe_pose[:3, :3]) @ curr_pose[:3, :3]
        trace = np.trace(R_diff)
        trace = min(3.0, max(-1.0, trace))
        delta_angle = np.arccos((trace - 1) / 2)

        return delta_trans > self.keyframe_dist_thresh or delta_angle > self.keyframe_angle_thresh

    def update(self, pcd_in_camera_frame, pose_world, force=False):
        if force or self.needs_update(pose_world):
            pcd_world = pcd_in_camera_frame.transform(pose_world)
            self.global_pcd += pcd_world
            self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size)
            self.global_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20)
            )
            self.last_keyframe_pose = pose_world
            return True
        return False

    def get_target(self):
        return self.global_pcd

# --- Data Loading ---
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, nargs='?', default='data/', help='Path to dataset folder (default: data/)')
    # [복구] every=5 (v3에서 잘 되던 값)
    parser.add_argument('--every', type=int, default=5)
    parser.add_argument('--voxel-size', type=float, default=0.015)
    parser.add_argument('--mesh-filename', type=str, default=None)
    parser.add_argument('--show-trajectory', action='store_true')
    parser.add_argument('--integrate', action='store_true')
    parser.add_argument('--point-clouds', action='store_true')
    return parser.parse_args()

def read_data(path: str) -> Dict:
    intrinsics_path = os.path.join(path, 'camera_matrix.csv')
    if not os.path.exists(intrinsics_path): return {}
    intrinsics_raw = np.loadtxt(intrinsics_path, delimiter=',')
    
    odom_path = os.path.join(path, 'odometry.csv')
    if os.path.exists(odom_path):
        odometry = np.loadtxt(odom_path, delimiter=',', skiprows=1)
        gt_poses = []
        gt_timestamps = []
        for line in odometry:
            t_wc = np.eye(4)
            t_wc[:3, 3] = line[2:5]
            t_wc[:3, :3] = Rotation.from_quat(line[5:]).as_matrix()
            gt_poses.append(t_wc)
            gt_timestamps.append(line[0])
    else:
        gt_poses, gt_timestamps = [], []

    depth_dir = os.path.join(path, 'depth')
    depth_frames = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(('.npy', '.png'))])

    return {'intrinsics_mat': intrinsics_raw, 'gt_poses': gt_poses, 'gt_timestamps': gt_timestamps, 'depth_frames': depth_frames}

def read_imu(path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    imu_path = os.path.join(path, 'imu.csv')
    if not os.path.exists(imu_path): return None
    data = np.loadtxt(imu_path, delimiter=',', skiprows=1)
    return data[:, 0], data[:, 1:4], data[:, 4:7]

def get_scaled_intrinsics(intrinsics_mat: np.ndarray) -> o3d.camera.PinholeCameraIntrinsic:
    scale_x = DEPTH_WIDTH / 1920
    scale_y = DEPTH_HEIGHT / 1440
    fx, fy = intrinsics_mat[0, 0] * scale_x, intrinsics_mat[1, 1] * scale_y
    cx, cy = intrinsics_mat[0, 2] * scale_x, intrinsics_mat[1, 2] * scale_y
    return o3d.camera.PinholeCameraIntrinsic(DEPTH_WIDTH, DEPTH_HEIGHT, fx, fy, cx, cy)

def load_depth_image(path: str) -> o3d.geometry.Image:
    if path.endswith('.npy'): depth_mm = np.load(path)
    else: depth_mm = np.array(Image.open(path))
    depth_m = depth_mm.astype(np.float32) / 1000.0
    return o3d.geometry.Image(depth_m)

# --- Pre-processing ---
def build_frame_pointclouds(args, data_dict: Dict) -> Tuple[List[o3d.geometry.PointCloud], List[int]]:
    intrinsics = get_scaled_intrinsics(data_dict['intrinsics_mat'])
    pc_list = []
    used_indices = []
    
    rgb_path = os.path.join(args.path, 'rgb.mp4')
    video = skvideo.io.vreader(rgb_path)
    depth_files = data_dict['depth_frames']
    
    print(">>> Building Point Clouds...")
    for i, rgb_frame in enumerate(video):
        if i >= len(depth_files): break
        if i % args.every != 0: continue
        
        depth_img = load_depth_image(depth_files[i])
        rgb_pil = Image.fromarray(rgb_frame).resize((DEPTH_WIDTH, DEPTH_HEIGHT))
        rgb_img = o3d.geometry.Image(np.array(rgb_pil))
        
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img, depth_img, depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False
        )
        pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)
        pc_list.append(pc)
        used_indices.append(i)
        print(f"   - Frame {i} processed", end='\r')
    print("\n>>> Point Cloud Generation Done.")
    return pc_list, used_indices

# --- IMU Processing (Rotation Only - Back to v3 Logic) ---
def estimate_gravity_alignment(accel: np.ndarray, samples: int = 50) -> np.ndarray:
    avg = np.mean(accel[:samples], axis=0)
    g_norm = np.linalg.norm(avg)
    g_meas = avg / g_norm
    g_target = np.array([0.0, 0.0, 1.0]) 

    v = np.cross(g_meas, g_target)
    c = np.dot(g_meas, g_target)
    s = np.linalg.norm(v)
    
    if s < 1e-6: return np.eye(3)
    
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R_init = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    return R_init

def integrate_gyro_rotation(timestamps: np.ndarray, accel: np.ndarray, gyro: np.ndarray) -> List[np.ndarray]:
    R_curr = estimate_gravity_alignment(accel)
    poses = []
    prev_t = timestamps[0]
    for i in range(len(timestamps)):
        dt = max(0, timestamps[i] - prev_t)
        if i > 0:
            w = gyro[i-1]
            angle = np.linalg.norm(w) * dt
            if angle > 1e-9:
                dR = Rotation.from_rotvec(w * dt).as_matrix()
                R_curr = R_curr @ dR
        T = np.eye(4)
        T[:3, :3] = R_curr
        poses.append(T)
        prev_t = timestamps[i]
    return poses

def sync_timestamps(cam_times: List[float], imu_times: np.ndarray, imu_data: List) -> List:
    synced = []
    n_imu = len(imu_times)
    for t in cam_times:
        idx = bisect_left(imu_times, t)
        if idx == 0: chosen = 0
        elif idx == n_imu: chosen = n_imu - 1
        else:
            chosen = idx - 1 if abs(imu_times[idx-1] - t) < abs(imu_times[idx] - t) else idx
        synced.append(imu_data[chosen])
    return synced

# --- Core SLAM (Refined v3 Logic) ---

def run_lio_pipeline(pc_list: List[o3d.geometry.PointCloud], imu_rotations: List[np.ndarray]) -> List[np.ndarray]:
    print("\n>>> Starting LIO Loop (Best Logic: IMU Rot + Map ICP)...")
    
    refined_poses = [np.eye(4)]
    global_map = GlobalMap(voxel_size=0.04)
    global_map.update(copy.deepcopy(pc_list[0]), refined_poses[0], force=True)
    
    loop_closed = False 
    estimation_method = o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.9)

    for k in range(1, len(pc_list)):
        print(f"   - Frame {k}/{len(pc_list)} | Map points: {len(global_map.get_target().points)}", end='\r')
        
        pc_curr = pc_list[k]
        
        # 1. Prediction (Rotation Only) - 이게 제일 안정적이었음
        R_prev = imu_rotations[k-1][:3, :3]
        R_curr = imu_rotations[k][:3, :3]
        R_diff = np.linalg.inv(R_prev) @ R_curr
        
        init_guess = np.eye(4)
        init_guess[:3, :3] = R_diff
        # Translation은 0으로 가정 (ICP가 다 찾음)
        
        T_prev = refined_poses[-1]
        T_predict = T_prev @ init_guess
        
        target_cloud = global_map.get_target()

        # [안전장치]
        if len(target_cloud.points) < 500:
            refined_poses.append(T_predict)
            global_map.update(copy.deepcopy(pc_curr), T_predict)
            continue

        # 2. Registration
        # Coarse (넓게 찾기)
        src_coarse = pc_curr.voxel_down_sample(0.1)
        src_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        reg_coarse = o3d.pipelines.registration.registration_icp(
            src_coarse, target_cloud, 
            0.5, # Threshold를 넓혀서(50cm) 빠른 이동 대응
            T_predict,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
        )
        
        # Fine (정밀하게 조이기) - Colored ICP
        src_fine = pc_curr.voxel_down_sample(0.04)
        src_fine.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        
        try:
            reg_fine = o3d.pipelines.registration.registration_icp(
                src_fine, target_cloud, 
                0.04, # Threshold를 좁혀서(4cm) 벽 두께 최소화 [중요]
                reg_coarse.transformation,
                estimation_method,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
            )
            T_curr = reg_fine.transformation
        except:
            T_curr = reg_coarse.transformation

        # 3. Loop Closure (Start point Snap)
        dist_to_start = np.linalg.norm(T_curr[:3, 3] - refined_poses[0][:3, 3])
        progress = k / len(pc_list)
        if not loop_closed and progress > 0.8 and dist_to_start < 1.5:
            start_cloud = pc_list[0].voxel_down_sample(0.04)
            start_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
            try:
                reg_loop = o3d.pipelines.registration.registration_icp(
                    src_fine, start_cloud, 0.5, T_curr,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
                )
                if reg_loop.fitness > 0.3:
                    print(f"\n[Success] Loop Closed! Fitness: {reg_loop.fitness:.2f}")
                    T_curr = reg_loop.transformation
                    loop_closed = True
            except: pass
            
        refined_poses.append(T_curr)
        global_map.update(copy.deepcopy(pc_curr), T_curr)

    print("\n>>> LIO Loop Completed.")
    
    # Post-processing
    print(">>> Removing Outliers...")
    final_map = global_map.get_target()
    cl, ind = final_map.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    global_map.global_pcd = final_map.select_by_index(ind)
    
    return refined_poses

# --- Visualization ---
def integrate_tsdf(args, data_dict, poses, used_indices) -> o3d.geometry.TriangleMesh:
    print(">>> Integrating Mesh (TSDF)...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size, sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intrinsics = get_scaled_intrinsics(data_dict['intrinsics_mat'])
    rgb_video = skvideo.io.vreader(os.path.join(args.path, 'rgb.mp4'))
    depth_files = data_dict['depth_frames']
    
    target_indices = set(used_indices[:len(poses)])
    frame_idx = 0
    pose_idx = 0
    
    for rgb_frame in rgb_video:
        if pose_idx >= len(poses): break
        if frame_idx in target_indices:
            depth_img = load_depth_image(depth_files[frame_idx])
            rgb_pil = Image.fromarray(rgb_frame).resize((DEPTH_WIDTH, DEPTH_HEIGHT))
            rgb_img = o3d.geometry.Image(np.array(rgb_pil))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_img, depth_img, depth_scale=1.0, depth_trunc=MAX_DEPTH, convert_rgb_to_intensity=False
            )
            volume.integrate(rgbd, intrinsics, np.linalg.inv(poses[pose_idx]))
            print(f"   - Integrated pose {pose_idx}/{len(poses)}", end="\r")
            pose_idx += 1
        frame_idx += 1
        
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print("\n>>> Mesh Integration Done.")
    return mesh

def save_poses_to_tum(poses: List[np.ndarray], timestamps: List[float], filename: str):
    print(f">>> Saving trajectory to {filename}...")
    with open(filename, 'w') as f:
        for pose, timestamp in zip(poses, timestamps):
            rotation_matrix = pose[:3, :3].copy()
            
            r = Rotation.from_matrix(rotation_matrix)
            
            q = r.as_quat() # (x, y, z, w)
            t = pose[:3, 3]
            
            f.write(f"{timestamp:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
    print(">>> Save done.")

def visualize(pc_list, poses, mesh=None, show_traj=False):
    print(">>> Visualizing...")
    geometries = []
    pc_global = o3d.geometry.PointCloud()
    for pc, T in zip(pc_list, poses):
        pc_copy = copy.deepcopy(pc)
        pc_copy.transform(T)
        pc_global += pc_copy
    geometries.append(pc_global.voxel_down_sample(0.02))
    
    if mesh: geometries.append(mesh)
    if show_traj:
        points = [T[:3, 3] for T in poses]
        lines = [[i, i+1] for i in range(len(points)-1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines)
        )
        line_set.paint_uniform_color([1, 0, 0]) 
        geometries.append(line_set)
    o3d.visualization.draw_geometries(geometries, window_name="Stray LIO Result")

# --- Main ---
def main():
    args = read_args()
    data = read_data(args.path)
    if not data['depth_frames']: return

    imu_result = read_imu(args.path)
    if imu_result is None: return
    imu_timestamps, imu_accel, imu_gyro = imu_result
    
    print(">>> Pre-integrating IMU Data (Rot Only)...")
    imu_rot_poses_all = integrate_gyro_rotation(imu_timestamps, imu_accel, imu_gyro)

    pc_list, used_indices = build_frame_pointclouds(args, data)

    cam_timestamps = []
    gt_len = len(data['gt_timestamps'])
    for idx in used_indices:
        if gt_len > 0:
            safe_idx = min(idx, gt_len - 1)
            cam_timestamps.append(data['gt_timestamps'][safe_idx])
        else:
            cam_timestamps.append(idx * 0.033)
    
    synced_imu_poses = sync_timestamps(cam_timestamps, imu_timestamps, imu_rot_poses_all)

    final_poses = run_lio_pipeline(pc_list, synced_imu_poses)

    # 첫 프레임(Identity)에 대한 타임스탬프가 리스트에 없을 수 있으므로 
    # cam_timestamps 앞에 첫 시간을 복사하거나, 그냥 저장할 때 개수를 맞춥니다.
    
    # 안전하게 cam_timestamps 개수에 맞춰서 저장
    save_poses_to_tum(final_poses, cam_timestamps, "my_trajectory.txt")

    mesh = None
    if args.mesh_filename or args.integrate:
        mesh = integrate_tsdf(args, data, final_poses, used_indices)
        if args.mesh_filename:
            o3d.io.write_triangle_mesh(args.mesh_filename, mesh)
            print(f"Saved mesh to {args.mesh_filename}")

    visualize(pc_list, final_poses, mesh, args.show_trajectory)

if __name__ == "__main__":
    main()