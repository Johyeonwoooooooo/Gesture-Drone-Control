import os
import shutil
import csv
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from argparse import ArgumentParser
from PIL import Image
import skvideo.io

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int

DEPTH_WIDTH = 256
DEPTH_HEIGHT = 192
MAX_DEPTH = 20.0


def read_args():
    parser = ArgumentParser()
    parser.add_argument("path", type=str, nargs='?', default="data/", help="Path to StrayScanner dataset folder")

    # 기본 동작
    parser.add_argument("--integrate", "-i", action="store_true", help="TSDF integrate and visualize mesh.")
    parser.add_argument("--point-clouds", "-p", action="store_true", help="Create merged point cloud and visualize.")
    parser.add_argument("--every", type=int, default=60, help="Sampling stride for point-cloud concatenation visualization.")

    # 저장 옵션
    parser.add_argument("--pc-filename", type=str, default=None, help="Save merged point cloud (.ply).")
    parser.add_argument("--mesh-filename", type=str, default=None, help="Save integrated mesh (.ply/.obj).")

    # Confidence / TSDF
    parser.add_argument("--confidence", "-c", type=int, default=1, help="Keep only depth with confidence >= level (0/1/2).")
    parser.add_argument("--voxel-size", type=float, default=0.015, help="TSDF voxel size (meters).")
    parser.add_argument("--sdf-trunc", type=float, default=0.05, help="TSDF sdf truncation (meters).")

    # PointCloud cleanup
    parser.add_argument("--pc-clean", action="store_true")
    parser.add_argument("--pc-voxel", type=float, default=0.02)
    parser.add_argument("--pc-stat-nb", type=int, default=30)
    parser.add_argument("--pc-stat-std", type=float, default=2.0)
    parser.add_argument("--pc-radius-nb", type=int, default=12)
    parser.add_argument("--pc-radius", type=float, default=0.06)
    parser.add_argument("--pc-keep-largest-cluster", action="store_true")
    parser.add_argument("--pc-cluster-eps", type=float, default=0.08)
    parser.add_argument("--pc-cluster-minpoints", type=int, default=30)

    # Mesh cleanup / smoothing
    parser.add_argument("--mesh-clean", action="store_true")
    parser.add_argument("--mesh-min-triangles", type=int, default=2000)
    parser.add_argument("--mesh-smooth", action="store_true")
    parser.add_argument("--mesh-smooth-iter", type=int, default=12, help="Lower default to avoid over-smoothing.")
    parser.add_argument("--mesh-smooth-lambda", type=float, default=0.3)
    parser.add_argument("--mesh-smooth-mu", type=float, default=-0.33)

    # refined dataset export (TSDF 결과를 프레임별 depth로 재렌더)
    parser.add_argument("--export-refined-dataset", type=str, default=None,
                        help="Output folder to write refined per-frame dataset (depth/rgb/intrinsics/odometry).")
    parser.add_argument("--export-rgb", action="store_true", help="Also export RGB frames as png (resized to depth resolution).")
    parser.add_argument("--depth-format", choices=["png16", "npy"], default="png16",
                        help="Depth output format. png16 writes millimeters uint16 PNG.")
    parser.add_argument("--max-depth", type=float, default=MAX_DEPTH, help="Max depth used for raycast output (meters).")

    # Open3D visualization tuning
    parser.add_argument("--viz", action="store_true", help="Visualize with tuned Open3D Visualizer (recommended).")
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--line-width", type=float, default=1.0)
    parser.add_argument("--show-back-face", action="store_true", help="Show back faces (sometimes reduces flicker).")

    return parser.parse_args()


def validate_dataset(path: str) -> bool:
    needed = ["rgb.mp4", "odometry.csv", "camera_matrix.csv", "depth"]
    for n in needed:
        if not os.path.exists(os.path.join(path, n)):
            print(f"[VALIDATE] Missing: {n}")
            return False
    return True


def _resize_camera_matrix(K, scale_x, scale_y):
    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]
    return np.array([
        [fx * scale_x, 0.0, cx * scale_x],
        [0.0, fy * scale_y, cy * scale_y],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def read_data(path: str):
    K = np.loadtxt(os.path.join(path, "camera_matrix.csv"), delimiter=",")
    odom = np.loadtxt(os.path.join(path, "odometry.csv"), delimiter=",", skiprows=1)

    poses = []
    for line in odom:
        # timestamp, frame, x,y,z, qx,qy,qz,qw
        pos = line[2:5]
        quat = line[5:]
        T_WC = np.eye(4, dtype=np.float64)
        T_WC[:3, :3] = Rotation.from_quat(quat).as_matrix()
        T_WC[:3, 3] = pos
        poses.append(T_WC)

    depth_dir = os.path.join(path, "depth")
    depth_frames = sorted([os.path.join(depth_dir, f) for f in os.listdir(depth_dir) if f.endswith(".npy") or f.endswith(".png")])

    return K, odom, poses, depth_frames


def load_confidence(path):
    return np.array(Image.open(path))


def load_depth_meters(depth_path, confidence=None, filter_level=0):
    if depth_path.endswith(".npy"):
        depth_mm = np.load(depth_path)
    elif depth_path.endswith(".png"):
        depth_mm = np.array(Image.open(depth_path))
    else:
        raise ValueError(f"Unsupported depth: {depth_path}")

    depth_m = depth_mm.astype(np.float32) / 1000.0
    if confidence is not None:
        depth_m[confidence < filter_level] = 0.0
    return depth_m


def get_intrinsics_for_depth(K_rgb):
    # StrayScanner camera_matrix.csv is typically for RGB full-res (ex: 1920x1440)
    K_d = _resize_camera_matrix(K_rgb, DEPTH_WIDTH / 1920.0, DEPTH_HEIGHT / 1440.0)
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
        width=DEPTH_WIDTH, height=DEPTH_HEIGHT,
        fx=K_d[0, 0], fy=K_d[1, 1], cx=K_d[0, 2], cy=K_d[1, 2]
    )
    return K_d, intrinsic_o3d


def clean_point_cloud(pc: o3d.geometry.PointCloud, flags):
    pc.remove_non_finite_points()

    if flags.pc_voxel and flags.pc_voxel > 0:
        pc = pc.voxel_down_sample(flags.pc_voxel)

    pc, _ = pc.remove_statistical_outlier(nb_neighbors=flags.pc_stat_nb, std_ratio=flags.pc_stat_std)
    pc, _ = pc.remove_radius_outlier(nb_points=flags.pc_radius_nb, radius=flags.pc_radius)

    if flags.pc_keep_largest_cluster and len(pc.points) > 0:
        labels = np.array(pc.cluster_dbscan(
            eps=flags.pc_cluster_eps,
            min_points=flags.pc_cluster_minpoints,
            print_progress=False
        ))
        if labels.size > 0 and labels.max() >= 0:
            valid = labels[labels >= 0]
            if valid.size > 0:
                largest_label = np.bincount(valid).argmax()
                keep_idx = np.where(labels == largest_label)[0]
                pc = pc.select_by_index(keep_idx)

    # 시각화 안정성을 위해 노멀도 계산
    if len(pc.points) > 1000:
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        try:
            pc.orient_normals_consistent_tangent_plane(50)
        except Exception:
            pass
    return pc


def clean_mesh(mesh: o3d.geometry.TriangleMesh, flags):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    # 데이터 경우에 따라 메쉬를 더 깨져 보이게 만들기도 하니까 주석 처리 했을 때와 안했을 때 비교 필요!!
    mesh.remove_non_manifold_edges()

    tri_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    if cluster_n_triangles.size > 0:
        remove = cluster_n_triangles[tri_clusters] < flags.mesh_min_triangles
        mesh.remove_triangles_by_mask(remove)
        mesh.remove_unreferenced_vertices()
    return mesh


def smooth_mesh(mesh: o3d.geometry.TriangleMesh, flags):
    mesh = mesh.filter_smooth_taubin(
        number_of_iterations=flags.mesh_smooth_iter,
        lambda_filter=flags.mesh_smooth_lambda,
        mu=flags.mesh_smooth_mu
    )
    return mesh


def build_tsdf_and_extract(path, poses, depth_frames, K_o3d, K_depth_3x3, flags):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=flags.voxel_size,
        sdf_trunc=flags.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    rgb_path = os.path.join(path, "rgb.mp4")
    video = skvideo.io.vreader(rgb_path)

    for i, (T_WC, rgb) in enumerate(zip(poses, video)):
        print(f"[TSDF] Integrating {i:06}", end="\r")

        # confidence
        conf_path = os.path.join(path, "confidence", f"{i:06}.png")
        confidence = load_confidence(conf_path) if os.path.exists(conf_path) else None

        depth_m = load_depth_meters(depth_frames[i], confidence, filter_level=flags.confidence)
        depth_img = o3d.geometry.Image(depth_m)

        # RGB -> depth resolution
        rgb_small = Image.fromarray(rgb).resize((DEPTH_WIDTH, DEPTH_HEIGHT), resample=Image.BILINEAR)
        rgb_small = np.asarray(rgb_small)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(rgb_small),
            depth_img,
            depth_scale=1.0,
            depth_trunc=flags.max_depth,
            convert_rgb_to_intensity=False
        )

        volume.integrate(rgbd, K_o3d, np.linalg.inv(T_WC))

    print("\n[TSDF] extract mesh ...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    if flags.mesh_clean:
        mesh = clean_mesh(mesh, flags)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

    if flags.mesh_smooth:
        mesh = smooth_mesh(mesh, flags)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()

    # TSDF point cloud 추출
    pc_from_tsdf = None
    try:
        pc_from_tsdf = volume.extract_point_cloud()
        if pc_from_tsdf is not None:
            pc_from_tsdf.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    except Exception:
        pc_from_tsdf = None

    return mesh, pc_from_tsdf


def render_depths_from_mesh(mesh_legacy, poses, K_depth_3x3, out_depth_dir,
                            depth_format="png16", max_depth=20.0):
    if not hasattr(o3d, "t"):
        raise RuntimeError("Open3D o3d.t (RaycastingScene) not available. Upgrade open3d.")

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    os.makedirs(out_depth_dir, exist_ok=True)

    intrinsic = o3d.core.Tensor(K_depth_3x3.astype(np.float64))
    width, height = DEPTH_WIDTH, DEPTH_HEIGHT

    for i, T_WC in enumerate(poses):
        if i % 50 == 0:
            print(f"[RAYCAST] depth {i:06}/{len(poses)}")

        # world->camera
        T_CW = np.linalg.inv(T_WC).astype(np.float64)
        extrinsic = o3d.core.Tensor(T_CW)

        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix=intrinsic,
            extrinsic_matrix=extrinsic,
            width_px=width,
            height_px=height
        )

        ans = scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy().astype(np.float64)  # (H,W) or (H*W,)

        rays_np = rays.numpy().reshape(-1, 6).astype(np.float64)  # (N,6)
        o = rays_np[:, 0:3]   # (N,3)
        d = rays_np[:, 3:6]   # (N,3)

        t = t_hit.reshape(-1)  # (N,)
        # 유효한 hit만 사용 (inf/nan/<=0/out of range 제거)
        valid = np.isfinite(t) & (t > 0.0) & (t < max_depth)

        depth_m = np.zeros((height * width,), dtype=np.float32)

        if np.any(valid):
            tv = t[valid][:, None]        # (Nv,1)
            ov = o[valid]                 # (Nv,3)
            dv = d[valid]                 # (Nv,3)

            # world hit points
            p_w = ov + tv * dv            # (Nv,3)

            R = T_CW[:3, :3]
            tt = T_CW[:3, 3]

            # camera points
            p_c = (p_w @ R.T) + tt        # (Nv,3)

            z = p_c[:, 2]                 # (Nv,)
            # 카메라 z가 정상 범위인 것만 depth로 채택
            z_valid = np.isfinite(z) & (z > 0.0) & (z < max_depth)

            idx = np.where(valid)[0]
            idx = idx[z_valid]

            depth_m[idx] = z[z_valid].astype(np.float32)

        depth_m = depth_m.reshape((height, width))

        if depth_format == "npy":
            np.save(os.path.join(out_depth_dir, f"{i:06}.npy"),
                    (depth_m * 1000.0).astype(np.float32))
        else:
            depth_mm = (depth_m * 1000.0).astype(np.uint16)
            Image.fromarray(depth_mm).save(os.path.join(out_depth_dir, f"{i:06}.png"))



def export_rgb_frames(path, out_rgb_dir):
    os.makedirs(out_rgb_dir, exist_ok=True)
    rgb_path = os.path.join(path, "rgb.mp4")
    video = skvideo.io.vreader(rgb_path)

    for i, rgb in enumerate(video):
        if i % 50 == 0:
            print(f"[RGB] export {i:06}")
        rgb_small = Image.fromarray(rgb).resize((DEPTH_WIDTH, DEPTH_HEIGHT), resample=Image.BILINEAR)
        rgb_small.save(os.path.join(out_rgb_dir, f"{i:06}.png"))


def write_odometry_csv(out_path, odom_array):
    # 원본 odometry.csv 포맷 유지: header + rows
    out_csv = os.path.join(out_path, "odometry.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for row in odom_array:
            w.writerow(row.tolist())


def write_camera_matrix_csv(out_path, K_3x3, filename="camera_matrix.csv"):
    np.savetxt(os.path.join(out_path, filename), K_3x3, delimiter=",")


def copy_if_exists(src, dst):
    if os.path.exists(src):
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


def export_refined_dataset(flags, mesh, odom, poses, K_depth_3x3):
    out_root = flags.export_refined_dataset
    os.makedirs(out_root, exist_ok=True)

    # 1) depth (re-rendered)
    out_depth_dir = os.path.join(out_root, "depth")
    render_depths_from_mesh(
        mesh_legacy=mesh,
        poses=poses,
        K_depth_3x3=K_depth_3x3,
        out_depth_dir=out_depth_dir,
        depth_format=flags.depth_format,
        max_depth=flags.max_depth
    )

    # 2) RGB frames
    if flags.export_rgb:
        out_rgb_dir = os.path.join(out_root, "rgb_frames")
        export_rgb_frames(flags.path, out_rgb_dir)

    # 3) odometry (우선 원래 odometry 유지, 추후 완성한 odometry 계산 알고리즘으로 치환해야함)
    write_odometry_csv(out_root, odom)

    # 4) intrinsics for exported resolution (depth/rgb_frames)
    write_camera_matrix_csv(out_root, K_depth_3x3, filename="camera_matrix.csv")

    # 5) 나머지 copy (IMU etc.)
    for fname in ["imu.csv", "imu.json", "accel.csv", "gyro.csv", "metadata.json"]:
        copy_if_exists(os.path.join(flags.path, fname), os.path.join(out_root, fname))

    copy_if_exists(os.path.join(flags.path, "confidence"), os.path.join(out_root, "confidence"))

    print(f"[EXPORT] refined dataset saved to: {out_root}")
    print("         - depth/ (re-rendered from TSDF mesh)")
    if flags.export_rgb:
        print("         - rgb_frames/ (resized to depth resolution)")
    print("         - odometry.csv (copied)")
    print("         - camera_matrix.csv (for 256x192 export resolution)")


def visualize_cloudcompare_like(geoms, flags):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D Viewer (tuned)", width=1280, height=720)

    for g in geoms:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = float(flags.point_size)
    opt.line_width = float(flags.line_width)
    opt.mesh_show_wireframe = False
    opt.mesh_show_back_face = bool(flags.show_back_face)
    opt.light_on = True

    vis.poll_events()
    vis.update_renderer()

    vis.run()
    vis.destroy_window()



def main():
    flags = read_args()

    if not validate_dataset(flags.path):
        print("[ERROR] Not a valid StrayScanner dataset folder.")
        return

    K_rgb, odom, poses, depth_frames = read_data(flags.path)
    K_depth_3x3, K_o3d = get_intrinsics_for_depth(K_rgb)

    # 기본 동작이 아무것도 없으면 integrate + export를 유도
    if not flags.integrate and not flags.point_clouds and flags.export_refined_dataset is None:
        flags.integrate = True

    geometries = []

    # merged point cloud (시각화/저장용)
    if flags.point_clouds:
        intrinsics = K_o3d
        pc = o3d.geometry.PointCloud()

        rgb_path = os.path.join(flags.path, "rgb.mp4")
        video = skvideo.io.vreader(rgb_path)

        for i, (T_WC, rgb) in enumerate(zip(poses, video)):
            if i % flags.every != 0:
                continue
            print(f"[PC] frame {i:06}", end="\r")

            T_CW = np.linalg.inv(T_WC)

            conf_path = os.path.join(flags.path, "confidence", f"{i:06}.png")
            confidence = load_confidence(conf_path) if os.path.exists(conf_path) else None

            depth_m = load_depth_meters(depth_frames[i], confidence, filter_level=flags.confidence)
            depth_img = o3d.geometry.Image(depth_m)

            rgb_small = Image.fromarray(rgb).resize((DEPTH_WIDTH, DEPTH_HEIGHT), resample=Image.BILINEAR)
            rgb_small = np.asarray(rgb_small)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_small),
                depth_img,
                depth_scale=1.0,
                depth_trunc=flags.max_depth,
                convert_rgb_to_intensity=False
            )

            pc += o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsic=T_CW)

        if flags.pc_clean:
            pc = clean_point_cloud(pc, flags)

        if flags.pc_filename is not None:
            o3d.io.write_point_cloud(flags.pc_filename, pc)
            print(f"\n[SAVED] point cloud -> {flags.pc_filename}")

        geometries.append(pc)

    # TSDF integrate -> mesh (+ export refined dataset)
    mesh = None
    pc_from_tsdf = None
    if flags.integrate or flags.export_refined_dataset is not None:
        mesh, pc_from_tsdf = build_tsdf_and_extract(flags.path, poses, depth_frames, K_o3d, K_depth_3x3, flags)

        if flags.mesh_filename is not None:
            o3d.io.write_triangle_mesh(flags.mesh_filename, mesh)
            print(f"[SAVED] mesh -> {flags.mesh_filename}")

        # refined dataset export
        if flags.export_refined_dataset is not None:
            export_refined_dataset(flags, mesh, odom, poses, K_depth_3x3)

        geometries.append(mesh)
        if pc_from_tsdf is not None and flags.pc_filename is not None and (not flags.point_clouds):
            # point_clouds 옵션 안 켰을 때도 TSDF pc를 pc-filename에 저장할 수 있게
            pc_to_save = pc_from_tsdf
            if flags.pc_clean:
                pc_to_save = clean_point_cloud(pc_to_save, flags)
            o3d.io.write_point_cloud(flags.pc_filename, pc_to_save)
            print(f"[SAVED] TSDF point cloud -> {flags.pc_filename}")

    # 시각화
    if flags.viz:
        visualize_cloudcompare_like(geometries, flags)
    else:
        o3d.visualization.draw_geometries(geometries)


if __name__ == "__main__":
    main()
