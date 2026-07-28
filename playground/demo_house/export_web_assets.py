# rooms_graph.json + npy 점군 -> 웹 UI용 경량 에셋 내보내기
#   <out>/floor_{f}.png   층별 톱다운 실색상 평면도 (천장 제거, 픽셀당 최저 z 점)
#   <out>/web_meta.json   층별 px<->world 매핑 + 방 bbox/center/passages + edges
#   <out>/points_xyz.f32  다운샘플 전역 점군 (float32 xyz interleaved)
#   <out>/points_rgb.u8   위 점군의 색 (uint8 RGB interleaved)
#   <out>/points_room.u8  위 점군의 방 인덱스 (meta['room_order'] 기준)
#
# 기본 출력은 이 폴더의 web_assets/ (파이프라인 산출물).
# 실사용 웹 앱이 서빙하는 사본으로 바로 내보내려면:
#   python export_web_assets.py --out ../../web/uploads
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
_ap = argparse.ArgumentParser(description='웹 UI용 경량 에셋 내보내기')
_ap.add_argument('--out', default=str(ROOT / 'web_assets'),
                 help='출력 폴더 (기본: 이 스크립트 옆의 web_assets/)')
OUT = Path(_ap.parse_args().out).resolve()
OUT.mkdir(parents=True, exist_ok=True)
print(f'[export] 출력 폴더: {OUT}')

RES = 0.02        # 평면도 해상도 (m/px)
CEIL_CUT = 0.80   # 방 z범위의 위쪽 20% (천장) 제거
VOXEL = 0.06      # 3D 점군 다운샘플 복셀 (m)

g = json.load(open(ROOT / 'rooms_graph.json', encoding='utf-8'))
rooms = g['rooms']

meta = {
    'scene': g['scene'],
    'res': RES,
    'coord_note': 'world(x,y,z): z=up. floor PNG: col=(x-xmin)/res, row=(ymax-y)/res',
    'floors': {},
    'rooms': {rid: {k: r.get(k) for k in ('floor', 'bbox_min', 'bbox_max', 'center', 'passages', 'label')}
              for rid, r in rooms.items()},
    'edges': g['edges'],
    'room_order': sorted(rooms),
}

by_floor = {}
for rid, r in rooms.items():
    by_floor.setdefault(r['floor'], []).append(rid)

xyz_all, rgb_all, ridx_all = [], [], []

for f, rids in sorted(by_floor.items()):
    mins = np.min([rooms[rid]['bbox_min'] for rid in rids], axis=0) - 0.3
    maxs = np.max([rooms[rid]['bbox_max'] for rid in rids], axis=0) + 0.3
    xmin, ymin, xmax, ymax = mins[0], mins[1], maxs[0], maxs[1]
    W, H = int(np.ceil((xmax - xmin) / RES)), int(np.ceil((ymax - ymin) / RES))
    img = np.zeros((H, W, 3), np.uint8)

    for rid in sorted(rids):
        d = ROOT / 'npy' / rooms[rid]['npy']
        c = np.load(d / 'coord.npy')
        col = np.load(d / 'color.npy').astype(np.uint8)

        # 3D 점군: 방 단위 복셀 다운샘플 (복셀당 첫 점)
        key = np.floor(c / VOXEL).astype(np.int64)
        _, keep = np.unique(key, axis=0, return_index=True)
        xyz_all.append(c[keep].astype(np.float32))
        rgb_all.append(col[keep])
        ridx_all.append(np.full(len(keep), meta['room_order'].index(rid), np.uint8))

        # 평면도: 천장 제거 후 낮은 z가 위에 오도록 기록
        z0, z1 = c[:, 2].min(), c[:, 2].max()
        m = c[:, 2] < z0 + CEIL_CUT * (z1 - z0)
        c2, col2 = c[m], col[m]
        order = np.argsort(-c2[:, 2])
        c2, col2 = c2[order], col2[order]
        px = np.clip(((c2[:, 0] - xmin) / RES).astype(int), 0, W - 1)
        py = np.clip(((ymax - c2[:, 1]) / RES).astype(int), 0, H - 1)
        img[py, px] = col2

    # 점 사이 빈 픽셀(검정) 채우기: 주변 3x3 최댓값으로 두 번 메꿈
    from scipy.ndimage import maximum_filter
    for _ in range(2):
        hole = img.max(axis=2) == 0
        filled = maximum_filter(img, size=(3, 3, 1))
        img[hole] = filled[hole]

    plt.imsave(OUT / f'floor_{f}.png', img)
    meta['floors'][str(f)] = {'xmin': float(xmin), 'ymin': float(ymin),
                              'xmax': float(xmax), 'ymax': float(ymax),
                              'width': W, 'height': H,
                              'png': f'floor_{f}.png', 'rooms': sorted(rids)}
    print(f'floor {f}: {W}x{H}px, rooms={sorted(rids)}')

xyz = np.vstack(xyz_all)
rgb = np.vstack(rgb_all)
ridx = np.concatenate(ridx_all)
xyz.tofile(OUT / 'points_xyz.f32')
rgb.tofile(OUT / 'points_rgb.u8')
ridx.tofile(OUT / 'points_room.u8')
meta['n_points_3d'] = int(len(xyz))

json.dump(meta, open(OUT / 'web_meta.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(f'3D points: {len(xyz):,} (voxel {VOXEL}m) -> '
      f'{(OUT / "points_xyz.f32").stat().st_size / 1e6:.1f}MB')
