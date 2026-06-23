# litept_indoor

실내 공간 포인트클라우드 → LitePT 시맨틱 세그멘테이션 → 물체 중심점 추출 → viser 시각화

## 파이프라인

```
coord.npy / color.npy / normal.npy
        ↓
  infer_centers.py   (LitePT semseg + DBSCAN 클러스터링)
        ↓
  centers.pkl        (per-point 라벨 + 인스턴스 중심점)
        ↓
  visualize.py       (viser 3D 시각화)
```

## 환경 설정

```bash
export PATH=$PATH:/home/minyoy/.local/bin:/usr/local/cuda-12.0/bin
export PYTHONPATH=/shareHost/minyoy/cumm:/shareHost/minyoy/spconv:$PYTHONPATH
export HF_HOME=/shareHost/minyoy/hf_cache
```

## 추론 실행

### 방 하나

```bash
cd /shareHost/minyoy/litept_indoor

/opt/conda/bin/python infer_centers.py \
    --npy_dir /shareHost/minyoy/project/data/npy/<ROOM_NAME>
```

### 전체 방 일괄

```bash
cd /shareHost/minyoy/litept_indoor

/opt/conda/bin/python infer_centers.py \
    --npy_root /shareHost/minyoy/project/data/npy
```

이미 `centers.pkl`이 존재하는 방은 건너뜁니다. 재추론하려면 `--skip_existing` 없이 실행하면 됩니다(기본값 False).

## 시각화 실행

```bash
cd /shareHost/minyoy/litept_indoor

/opt/conda/bin/python visualize.py \
    --npy_root /shareHost/minyoy/project/data/npy
```

브라우저에서 **http://localhost:8080** 접속

### 옵션

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--npy_root` | — | 전체 방 디렉토리 |
| `--pkl` | — | 방 하나의 centers.pkl 경로 |
| `--port` | 8080 | viser 서버 포트 |
| `--max_pts` | 200000 | 방당 최대 표시 포인트 수 |

### GUI 패널

| 폴더 | 컨트롤 | 설명 |
|---|---|---|
| Point Cloud | Color Mode | Label / RGB / Blend 전환 |
| Point Cloud | Blend (label %) | Blend 모드에서 라벨 색 비율 |
| Point Cloud | Point Size | 포인트 크기 |
| DBSCAN | eps (m) | 클러스터 반경 — 낮을수록 엄격 |
| DBSCAN | min_samples | 최소 포인트 수 — 높을수록 작은 클러스터 제거 |
| DBSCAN | Apply DBSCAN | 현재 파라미터로 중심점 재계산 |
| Labels | Show center labels | 중심점 위 클래스 이름 표시 토글 |

## 입력 데이터 형식

```
data/npy/<ROOM_NAME>/
    coord.npy    # (N, 3) float32, 세계 좌표 (m)
    color.npy    # (N, 3) uint8 or float, RGB
    normal.npy   # (N, 3) float32, 법선 벡터
```

## 출력

```
data/npy/<ROOM_NAME>/centers.pkl
```

```python
{
    'room':        str,               # 방 이름
    'classes':     list[str],         # ScanNet 20개 클래스
    'pred_labels': np.ndarray (N,),   # per-point 라벨 (-1: 미분류)
    'centers':     list[dict],        # 인스턴스 중심점 목록
    'coord':       np.ndarray (N, 3), # 원본 세계 좌표
}
```

각 인스턴스 dict:

```python
{
    'class_name':    str,
    'label_idx':     int,
    'center':        np.ndarray (3,),  # 세계 좌표 중심점
    'point_indices': np.ndarray,
    'n_points':      int,
}
```
