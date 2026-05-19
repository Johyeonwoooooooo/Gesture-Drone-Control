# convert.py 수정
import numpy as np

COORD  = '/shareHost/minyoy/project/data/coord.npy'
COLOR  = '/shareHost/minyoy/project/data/color.npy'
NORMAL = '/shareHost/minyoy/project/data/normal.npy'
OUT    = '/shareHost/minyoy/project/data/my_scene.bin'

VOXEL_SIZE = 0.05  # 5cm voxel — 너무 많으면 0.1로 키우세요

def main():
    coord  = np.load(COORD).astype(np.float32)
    color  = np.load(COLOR).astype(np.float32)
    normal = np.load(NORMAL).astype(np.float32)

    if color.max() > 1.0:
        color = color / 127.5 - 1.0

    # voxel 다운샘플링
    voxel_idx = np.floor(coord / VOXEL_SIZE).astype(np.int64)
    _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
    
    coord  = coord[unique_idx]
    color  = color[unique_idx]
    normal = normal[unique_idx]
    print(f'다운샘플링 후 점 수: {len(coord)}')

    points = np.concatenate([coord, color, normal], axis=1).astype(np.float32)
    points.tofile(OUT)
    print(f'저장 완료 → {OUT}')

if __name__ == '__main__':
    main()
