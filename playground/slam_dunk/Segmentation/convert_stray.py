import os
import numpy as np
import open3d as o3d
import shutil
from pathlib import Path

def convert_stray_to_mosaic_npy(source_path, target_root, scan_id):
    """
    StrayScanner의 ply -> Mosaic3D용 coord.npy, color.npy 분리 저장
    """
    ply_path = os.path.join(source_path, "pointcloud_refined.ply")
    
    if not os.path.exists(ply_path):
        print(f"[Skip] {scan_id}: pointcloud_refined.ply 없음")
        return False

    # 1. Point Cloud 읽기
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) # 0.0 ~ 1.0 범위

    # 2. 좌표계 변환 (StrayScanner Y-up -> Mosaic3D Z-up)
    # x, y, z -> x, -z, y 로 변환하여 바닥을 평평하게 맞춤
    R = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, 1, 0]])
    points = points @ R.T

    # 3. 저장할 폴더 생성 (dataset_base.py는 폴더 단위로 읽음)
    # 예: /datasets/.../arkitscenes/my_scan_01/
    scan_dir = os.path.join(target_root, scan_id)
    os.makedirs(scan_dir, exist_ok=True)

    # 4. coord.npy 저장 (Float32)
    np.save(os.path.join(scan_dir, "coord.npy"), points.astype(np.float32))

    # 5. color.npy 저장 (Range 0~255, Float or Int)
    # Mosaic3D 코드는 보통 0~255 범위를 기대하므로 변환
    colors_255 = (colors * 255).astype(np.uint8) # 용량을 위해 uint8 추천
    np.save(os.path.join(scan_dir, "color.npy"), colors_255)

    print(f"[Done] {scan_id} 변환 완료 -> {scan_dir}")
    return True

def main():
    # ================= [경로 수정 완료] =================
    
    # 1. Source: 내 SLAM 결과물 (Stray_refined 폴더)
    # 스크립트 위치를 기준으로 워크스페이스/리포지토리 상대 경로를 사용
    repo_root = Path(__file__).resolve().parents[1]  # Lidar-IMU-SLAM-DUNK
    workspace_root = repo_root.parent  # 상위 작업 폴더 (예: slamdunk)

    # 이미지 image_86bccd.png 참고
    source_root = str(workspace_root / "Stray_refined")

    # 2. Target: Mosaic3D가 읽을 데이터셋 폴더
    # 이미지 image_86b8ef.png 참고 (datasets/mosaic3d/data 밑에 arkitscenes 생성)
    target_root = str(workspace_root / "datasets" / "mosaic3d" / "data" / "arkitscenes")

    # 3. Split File: 'hr' 작업 폴더 내의 split_files 위치
    # 이미지 image_86b8b1.png 참고
    split_file_dir = str(repo_root / "Mosaic3D" / "src" / "data" / "metadata" / "split_files")
    
    # ====================================================

    # 폴더 자동 생성 (없으면 만듦)
    os.makedirs(target_root, exist_ok=True)
    os.makedirs(split_file_dir, exist_ok=True)

    # source_root 안의 폴더들을 스캔 (예: refined_dataset_0113)
    subfolders = [f.path for f in os.scandir(source_root) if f.is_dir()]
    valid_scans = []

    print(f"총 {len(subfolders)}개의 스캔 발견. 변환 시작...")

    for folder in subfolders:
        scan_id = os.path.basename(folder)
        success = convert_stray_to_mosaic_npy(folder, target_root, scan_id)
        if success:
            valid_scans.append(scan_id)

    # 6. Split 파일 생성 (dataset_base.py가 이 파일을 보고 데이터를 로드함)
    # 파일명 규칙: {dataset_name}_{split}.txt
    # data_dir이 'arkitscenes'로 끝나므로 dataset_name은 'arkitscenes'가 됨
    split_name = "arkitscenes_val.txt" 
    split_path = os.path.join(split_file_dir, split_name)
    
    with open(split_path, "w") as f:
        for scan_id in valid_scans:
            f.write(f"{scan_id}\n")
            
    print("------------------------------------------------")
    print(f"변환 끝! 총 {len(valid_scans)}개 데이터 처리됨.")
    print(f"데이터 위치: {target_root}")
    print(f"리스트 파일: {split_path}")
    print("이제 ar_only.yaml에서 split: val 로 설정하고 돌리시면 됩니다.")

if __name__ == "__main__":
    main()