# 3D-segmentation: Mosaic3D open-vocab viewer

Mosaic3D (SpUNet101 + CLIP text head) 추론 결과를 viser 웹앱으로 자연어 쿼리하며 들여다보는 파이프라인.

## 0. 사전 요구사항

| 항목 | 위치 / 값 |
|---|---|
| GPU | RTX 3080급 1장 (8GB+) |
| Mosaic3D 레포 | `../Mosaic3D/` (프로젝트 루트와 같은 레벨) |
| 모델 ckpt | `../data/spunet101.ckpt` |
| 데이터 (옵션 1) | `../data/matterport3d_compressed/<houseID>_NN/{coord,color,normal,segment}.npy` |
| 데이터 (옵션 2) | HM3D `.glb` (GitHub example tar에서 받음) |
| conda | `/data1/workspaces/jgshin22/miniconda3` 가정 (다른 위치면 `setup_env.sh` 수정) |
| tmux | 시스템에 설치돼 있어야 함 |

## 1. 환경 설정 (한 번만)

`mosaic3d` conda env 생성 — torch 2.2.2 + cu121 + spconv-cu120 + open_clip + viser. 약 5~10분.

```bash
cd /data1/workspaces/jgshin22/Gesture-Drone-Control
tmux new -s mosaic-env -d \
  'bash 3D-segmentation/setup_env/setup_env.sh 2>&1 | tee 3D-segmentation/setup_env/setup.log'
# 진행상황: tmux attach -t mosaic-env  (Ctrl+b d 로 나가기)
```

성공하면 마지막에 `[setup] DONE` 출력.

> **주의**: `transformers<5`, `setuptools<81` 핀이 들어있음. lightning 2.2가 `pkg_resources`를 import 하기 때문.

## 2. 입력 데이터 준비

### 옵션 A: Pointcept compressed Matterport3D (이미 있음)
`../data/matterport3d_compressed/` 에 90 house × 평균 24 region이 들어있음. 추가 작업 불필요.

### 옵션 B: HM3D example mesh
TOS 없이 받을 수 있는 HM3D 예시 3개 scene:

```bash
cd 3D-segmentation/cache
mkdir -p hm3d_example && cd hm3d_example
wget -O hm3d-example-glb-v0.2.tar \
  https://github.com/matterport/habitat-matterport-3dresearch/raw/main/example/hm3d-example-glb-v0.2.tar
tar xf hm3d-example-glb-v0.2.tar
# 추출되는 scene: 00337-CFVBbU9Rsyb, 00770-NBg5UqG3di3, 00861-GLAQ4DNUx5U
```

`.glb` → compressed 포맷(.npy)으로 변환:

```bash
cd /data1/workspaces/jgshin22/Gesture-Drone-Control/3D-segmentation
source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh
conda activate mosaic3d

python scripts/prepare_hm3d.py \
  --glb cache/hm3d_example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.glb \
  --scene-id GLAQ4DNUx5U \
  --out-dir cache/hm3d_compressed
```

원본 mesh는 별도로 `cache/hm3d_compressed/<scene>_mesh/` 에 vertex/face/color로 저장.

## 3. 추론 (per-point CLIP feature 캐시 생성)

### Compressed MP3D 한 house 전체:
```bash
tmux new -s mosaic-precompute -d \
  'bash 3D-segmentation/scripts/run_precompute.sh 17DRP5sb8fy cuda:0 \
   2>&1 | tee 3D-segmentation/cache/precompute.log'
```
약 8초, region 10개, 산출 약 2.2GB.

### HM3D scene 1개:
```bash
conda activate mosaic3d
cd 3D-segmentation
python inference/run_inference.py \
  --ckpt /data1/workspaces/jgshin22/Gesture-Drone-Control/data/spunet101.ckpt \
  --data-dir cache/hm3d_compressed \
  --out-dir cache/feat_hm3d \
  --regions GLAQ4DNUx5U \
  --device cuda:0
```

HM3D scene을 webapp의 region 드롭다운에 노출시키려면 feat과 mesh를 `cache/feat`, `cache/match`로 연결:

```bash
python - <<'PY'
import numpy as np, os
from pathlib import Path
scene = "GLAQ4DNUx5U"
root = Path("/data1/workspaces/jgshin22/Gesture-Drone-Control/3D-segmentation")
src = root / "cache" / "feat_hm3d" / scene
dst = root / "cache" / "feat" / scene
dst.mkdir(parents=True, exist_ok=True)
for f in ["feat.npy", "coord.npy"]:
    t = dst / f
    if t.exists() or t.is_symlink(): t.unlink()
    t.symlink_to(src / f)
mesh = root / "cache" / "hm3d_compressed" / f"{scene}_mesh"
match = root / "cache" / "match" / scene
match.mkdir(parents=True, exist_ok=True)
verts = np.load(mesh / "vertices.npy")
np.save(match / "vertices.npy", verts)
np.save(match / "faces.npy", np.load(mesh / "faces.npy"))
np.save(match / "vertex_colors.npy", np.load(mesh / "vertex_colors.npy"))
np.save(match / "vertex2point.npy", np.arange(len(verts), dtype=np.int32))
np.save(match / "vertex2dist.npy", np.zeros(len(verts), dtype=np.float32))
print("linked")
PY
```

## 4. 웹앱 실행

```bash
tmux new -s mosaic-web -d \
  "source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh && \
   conda activate mosaic3d && \
   cd /data1/workspaces/jgshin22/Gesture-Drone-Control/3D-segmentation && \
   python webapp/server.py --port 8080 --host 0.0.0.0 \
   2>&1 | tee cache/webapp.log"
```

20초 정도 후 `http://<host>:8080` 접속 (CLIP 모델 로딩에 시간 걸림).

GUI 컨트롤:
- **Backend**: `mosaic3d`(heatmap+DBSCAN) / `unidet3d`(3D detection+CLIP 매칭).
  `unidet3d` 옵션은 `--enable-unidet3d` 로 켜고 `unidet3d` env 에서 서버를 띄울 때만 보임
  (자세한 건 [`webapp_llm/README.md`](webapp_llm/README.md) "UniDet3D 백엔드 모드").
- **Region**: 추론 캐시가 있는 scene 드롭다운
- **Mode**: `rgb` / `single query` / `class list` (mosaic3d 백엔드)
- **Query / classes**: 단일 프롬프트 또는 콤마 구분 클래스 리스트
- **Threshold**: single query에서 cosine sim 컷오프
- **Point size**: 점 크기
- **Query opacity**: heatmap을 RGB와 블렌드 (1=쿼리만, 0=RGB만, 0.7 권장)
- **UniDet3D: score thr / top-K / show all / (re)run**: unidet3d 백엔드 컨트롤
- **Apply**: 적용

마우스:
- 좌클릭 드래그 → 회전
- 우클릭 드래그 → 평행이동(pan)
- 휠 → 줌

## 5. 디렉토리 구조

```
3D-segmentation/
├── README.md
├── setup_env/
│   ├── setup_env.sh                # conda env 생성 스크립트
│   ├── requirements-inference.txt  # 핀된 deps
│   └── setup.log                   # 마지막 설치 로그
├── inference/
│   └── run_inference.py            # ckpt + region.npy → per-point feat .npy
├── matching/
│   └── align_to_mesh.py            # compressed coord ↔ 원본 region.ply (MP3D 원본 받았을 때)
├── scripts/
│   ├── prepare_hm3d.py             # .glb → coord/color/normal/segment .npy
│   └── run_precompute.sh           # tmux용 일괄 추론 래퍼
├── webapp/
│   └── server.py                   # viser 서버
└── cache/
    ├── feat/<region>/              # ← webapp이 읽음 (feat.npy, coord.npy)
    ├── feat_hm3d/<scene>/          # HM3D 추론 raw (cache/feat에 symlink)
    ├── hm3d_compressed/<scene>/    # prepare_hm3d.py 출력 (compressed 포맷)
    ├── hm3d_compressed/<scene>_mesh/  # 원본 mesh vertex/face/colors
    ├── hm3d_example/               # 다운받은 .glb tar 추출본
    ├── match/<region>/             # ← webapp이 읽음 (vertex_colors.npy 등)
    ├── precompute.log
    └── webapp.log
```

## 6. tmux 세션 관리

| 세션 | 역할 |
|---|---|
| `mosaic-env` | env 설치 (1회성) |
| `mosaic-precompute` | feat 캐시 생성 (region 추가시마다) |
| `mosaic-web` | viser 서버 (계속 떠 있음) |

```bash
tmux ls                       # 세션 목록
tmux attach -t mosaic-web     # 들어가기 (Ctrl+b d 로 나오기)
tmux kill-session -t mosaic-web  # 정지
```

## 7. 새 region/scene 추가 방법

1. **MP3D compressed에서**: `bash scripts/run_precompute.sh <houseID>` → 자동으로 `cache/feat/<region>/` 채움
2. **HM3D .glb에서**: `prepare_hm3d.py` → `run_inference.py` → 위 4단계의 symlink 블록
3. webapp 새로고침하면 드롭다운에 자동 반영 (서버 재시작 불필요... 는 아니고 현재 코드는 시작 시 한 번만 스캔하므로 재시작 필요. 빠르게 재시작: `tmux kill-session -t mosaic-web` 후 4단계 명령 다시)

## 8. 트러블슈팅

| 증상 | 원인/해결 |
|---|---|
| `ModuleNotFoundError: pkg_resources` | `pip install 'setuptools<81'` |
| `transformers` 가 torch 2.2 거부 | `pip install 'transformers<5'` |
| viser 점 색이 회색 | colors가 uint8이면 일부 빌드에서 잘못 해석. `server.py`는 float32 [0,1]로 변환해 보냄 (이미 적용) |
| GPU OOM | `--device cuda:1` 등으로 다른 GPU; 큰 region(>50만 점)은 region 분할 필요 |
| `viser` connection 안 됨 | 방화벽/포트 확인. `--host 0.0.0.0` 사용 |
| 좌표가 화면 밖 | `server.py`가 자동 centering. 그래도 멀면 region 다시 선택 |

## 9. 모델/데이터 요약

- **모델**: SparseUNet-101, Mosaic3D 학습 (ScanNet + ARKitScenes + ScanNetPP), out_dim=768 (CLIP ViT-L 텍스트 차원)
- **PPT condition**: `ScanNet` 고정 (Matterport는 PPT에 미포함, 가장 가까운 indoor 컨디션)
- **CLIP 텍스트 인코더**: `hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP` (open_clip)
- **추론 입력**: coord (CenterShift apply_z=True), color/127.5-1, grid_size=0.02
- **추론 출력**: per-point feature (N, 768) fp16

쿼리 시 텍스트는 동일 CLIP 인코더로 임베딩해 cosine similarity 계산. argmax(class list) 또는 threshold/percentile-heatmap(single query).
