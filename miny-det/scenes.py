# scenes.py — 공통 씬 목록 + 경로 헬퍼 (convert/infer/clip_index/app 가 공유)
import os
import glob

# hm3d_compressed val split 루트 (coord/color/normal.npy 가 들어있는 곳)
SRC_ROOT = '/home/jgshin22/work/Gesture-Drone-Control/data/hm3d_compressed/val'

# miny-det/data — 이 파일 기준 상대경로로 고정 (어디서 실행해도 동일)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

# 전처리 대상: 00800_TEEsavR23oF, 00809_Qpor2mEya8F 의 모든 scene
REGIONS = [
    '00800_TEEsavR23oF_000_002',
    '00800_TEEsavR23oF_000_003',
    '00800_TEEsavR23oF_000_004',
    '00800_TEEsavR23oF_000_005',
    '00800_TEEsavR23oF_000_006',
    '00800_TEEsavR23oF_000_007',
    '00800_TEEsavR23oF_000_008',
    '00800_TEEsavR23oF_000_009',
    '00800_TEEsavR23oF_001_010',
    '00800_TEEsavR23oF_001_011',
    '00809_Qpor2mEya8F_000_002',
    '00809_Qpor2mEya8F_000_003',
    '00809_Qpor2mEya8F_001_004',
    '00809_Qpor2mEya8F_001_005',
    '00809_Qpor2mEya8F_001_006',
    '00809_Qpor2mEya8F_001_007',
    '00809_Qpor2mEya8F_001_008',
    '00809_Qpor2mEya8F_001_009',
    '00809_Qpor2mEya8F_001_010',
    '00809_Qpor2mEya8F_002_011',
    '00809_Qpor2mEya8F_002_012',
    '00809_Qpor2mEya8F_002_013',
    '00809_Qpor2mEya8F_002_014',
    '00809_Qpor2mEya8F_002_015',
    '00809_Qpor2mEya8F_002_016',
    '00809_Qpor2mEya8F_002_017',
    '00809_Qpor2mEya8F_002_018',
    '00809_Qpor2mEya8F_002_019',
    '00809_Qpor2mEya8F_002_020',
    '00809_Qpor2mEya8F_002_021',
    '00809_Qpor2mEya8F_002_022',
    '00809_Qpor2mEya8F_002_023',
]


def src_npy(region, kind):
    """kind: 'coord' | 'color' | 'normal'"""
    return os.path.join(SRC_ROOT, region, f'{kind}.npy')


def bin_path(region):
    return os.path.join(DATA_DIR, f'{region}.bin')


def det_path(region):
    return os.path.join(DATA_DIR, f'det_{region}.pkl')


def index_path(region):
    return os.path.join(DATA_DIR, f'det_{region}_clip_index.pkl')


def available_regions():
    """이미 clip_index 까지 만들어진 region 들 (app.py 드롭다운 소스)."""
    found = []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, 'det_*_clip_index.pkl'))):
        name = os.path.basename(p)
        region = name[len('det_'):-len('_clip_index.pkl')]
        found.append(region)
    return found
