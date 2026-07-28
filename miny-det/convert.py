# convert.py — coord/color/normal.npy → (N,9) .bin (voxel 다운샘플), 모든 REGION 일괄
import sys
import numpy as np

import scenes

VOXEL_SIZE = 0.01  # 1cm voxel — 너무 많으면 0.1로 키우세요


def convert_region(region):
    coord = np.load(scenes.src_npy(region, 'coord')).astype(np.float32)
    color = np.load(scenes.src_npy(region, 'color')).astype(np.float32)
    normal = np.load(scenes.src_npy(region, 'normal')).astype(np.float32)

    if color.max() > 1.0:
        color = color / 127.5 - 1.0

    # voxel 다운샘플링
    voxel_idx = np.floor(coord / VOXEL_SIZE).astype(np.int64)
    _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)

    coord = coord[unique_idx]
    color = color[unique_idx]
    normal = normal[unique_idx]

    points = np.concatenate([coord, color, normal], axis=1).astype(np.float32)
    out = scenes.bin_path(region)
    points.tofile(out)
    print(f'[convert] {region}: {len(coord)} pts → {out}')


def main():
    regions = sys.argv[1:] or scenes.REGIONS
    for i, region in enumerate(regions):
        print(f'=== ({i + 1}/{len(regions)}) {region} ===')
        convert_region(region)
    print('[convert] done.')


if __name__ == '__main__':
    main()
