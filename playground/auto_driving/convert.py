"""
[1단계] GLB → .npy 변환 저장
한 번만 실행하면 됨. 결과를 house_voxel.npy로 저장.
"""

import numpy as np
import trimesh
import sys
import os

VOXEL_SIZE = 0.1

def load_glb(path):
    print(f"[1/3] GLB 로딩 중: {path}")
    scene = trimesh.load(path)
    if isinstance(scene, trimesh.Scene):
        meshes = [g for g in scene.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("GLB 안에 mesh가 없습니다.")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = scene
    print(f"    vertices: {len(mesh.vertices):,}, faces: {len(mesh.faces):,}")
    return mesh

def make_points(mesh, pitch=0.1):
    print(f"[2/3] Voxel 변환 중 (pitch={pitch}m) ...")
    voxel_grid = mesh.voxelized(pitch=pitch)
    occupied   = voxel_grid.matrix.copy()

    # pitch / origin 추출 (trimesh 버전 호환)
    if hasattr(voxel_grid, 'pitch'):
        p = float(np.atleast_1d(voxel_grid.pitch)[0])
    elif hasattr(voxel_grid, 'scale'):
        p = float(np.atleast_1d(voxel_grid.scale)[0])
    else:
        p = pitch

    if hasattr(voxel_grid, 'transform'):
        origin = np.array(voxel_grid.transform[:3, 3], dtype=float)
    else:
        origin = voxel_grid.points.min(axis=0)

    # 천장 상위 3레이어 제거 (interior 모드)
    occupied[:, :, -3:] = False

    obs_idx      = np.argwhere(occupied)
    obstacle_pts = (obs_idx * p + origin).astype(np.float32)

    print(f"    포인트 수: {len(obstacle_pts):,}")
    return obstacle_pts

def main():
    glb_path = sys.argv[1] if len(sys.argv) > 1 else "house.glb"
    if not os.path.exists(glb_path):
        print(f"[오류] 파일 없음: {glb_path}")
        sys.exit(1)

    mesh = load_glb(glb_path)
    pts  = make_points(mesh, pitch=VOXEL_SIZE)

    # 저장 경로: GLB와 같은 폴더, 같은 이름으로 .npy
    out_path = os.path.splitext(glb_path)[0] + "_voxel.npy"
    np.save(out_path, pts)
    print(f"[3/3] 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
