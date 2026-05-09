import argparse
import numpy as np
import open3d as o3d
import os

def main():
    parser = argparse.ArgumentParser(description='Convert HM3D npy files to PLY point cloud')
    parser.add_argument('data_dir', help='디렉토리 경로 (coord.npy, color.npy 포함)')
    parser.add_argument('--output', '-o', default=None, help='출력 PLY 파일 경로 (기본: data_dir/scene.ply)')
    args = parser.parse_args()

    coord_path = os.path.join(args.data_dir, 'coord.npy')
    color_path = os.path.join(args.data_dir, 'color.npy')

    assert os.path.exists(coord_path), f'coord.npy 없음: {coord_path}'
    assert os.path.exists(color_path), f'color.npy 없음: {color_path}'

    print(f'로드 중: {args.data_dir}')
    coord = np.load(coord_path)
    color = np.load(color_path)
    print(f'포인트 수: {len(coord):,}')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(color.astype(np.float64) / 255.0)

    if args.output is None:
        scene_name = os.path.basename(args.data_dir.rstrip('/'))
        out_path = os.path.join(args.data_dir, f'{scene_name}.ply')
    else:
        out_path = args.output

    o3d.io.write_point_cloud(out_path, pcd)
    print(f'저장 완료: {out_path}')

if __name__ == '__main__':
    main()
