# webapp_llm — Natural-language → 3D segmentation

기존 [`webapp/`](../webapp/) 위에 **local LLM 의도 파싱**을 얹은 별도 viser 웹앱입니다.
기존 웹앱은 수정하지 않고, 그 안의 로딩/클러스터링 유틸만 import 해서 재사용합니다.

## 파이프라인

```
자연어 명령 (예: "위층 방의 화장실 사진 촬영해줘")
        │
        ▼
  Local LLM (Qwen2.5-3B-Instruct, 기본)
        │  → JSON {target_object, clip_prompt, location_hint, action, return_home}
        ▼
  CLIP 텍스트 인코딩 (clip_prompt)
        │
        ▼
  점별 cosine similarity heatmap  →  DBSCAN 클러스터링
        │
        ▼
  Top-K 3D 후보 (rank, world center, bbox, score)  → viser 시각화
```

## 실행

```bash
conda activate mosaic3d   # 기존 inference 환경 그대로 사용
pip install "transformers>=4.45" "accelerate>=0.34"   # 처음 한 번만

python 3D-segmentation/webapp_llm/server.py \
    --port 8090 \
    --llm-model Qwen/Qwen2.5-3B-Instruct \
    --llm-device cuda:1 \
    --clip-device cuda:0
```

브라우저에서 `http://<host>:8090` 접속 후

1. Building / Room 드롭다운으로 씬을 선택
2. "Drone command" 텍스트박스에 자연어 명령 입력
3. **Parse + Localize** 버튼

→ Parsed intent + 3D 후보(World 좌표 포함)가 패널에 표시되고, scene에는 히트맵과
후보별 sphere/bbox/label 마커가 렌더링됩니다.

## 8× RTX 3080 (8GB) 권장 모델 매핑

| 모델 | 권장 옵션 |
| --- | --- |
| `Qwen/Qwen2.5-3B-Instruct` (기본) | `--llm-device cuda:1` — 1장에 fp16으로 적재 |
| `Qwen/Qwen2.5-7B-Instruct` | `--llm-device-map auto` — 여러 GPU에 샤딩 |
| `meta-llama/Llama-3.2-3B-Instruct` | `--llm-device cuda:1` |
| INT4 양자화 모델 (e.g. AWQ) | `--llm-dtype float16 --llm-device cuda:1` |

CLIP은 1장(`--clip-device cuda:0`)에 두고 LLM은 별도 GPU(`cuda:1`)에 두면
서로 간섭하지 않습니다.

## 출력 좌표

후보의 `center` 는 viser 디스플레이용으로 씬 bbox 중심을 뺀 좌표입니다.
패널의 `world=(x,y,z)` 가 원본 월드 좌표 (= `asset.center + display_center`)
이며, 이 값을 path planner에 그대로 넘기면 됩니다.

## UniDet3D 백엔드 모드 (옵션)

기본 heatmap+DBSCAN 경로 외에, **UniDet3D 3D object detection → CLIP bbox 매칭**
경로를 GUI 상단의 **`Backend` 드롭다운**(`mosaic3d` ↔ `unidet3d`)으로 골라 쓸 수 있습니다.
별도 `.bin` 을 받지 않고 **현재 로드된 cache 씬**(`asset.coord` + `vertex_colors`)을
그대로 detector 에 넣습니다 — 즉 mosaic3d 와 UniDet3D 가 같은 점군 위에서 동작합니다.
이 백엔드는 base `webapp/server.py` 에도 동일하게 들어가 있습니다 (공유 모듈
[`webapp/unidet3d_backend.py`](../webapp/unidet3d_backend.py)).

```
현재 로드된 cache 씬 (xyz + rgb)
        │
        ▼
  UniDet3D (multi-head: ScanNet++/ScanNet/S3DIS/...)
        │  → bboxes / labels / scores
        ▼
  CLIP 클래스 임베딩 (webapp TextEncoder 공유) →  (LLM이 뽑은) clip_prompt 와 cosine sim
        │
        ▼
  Top-K bbox (★ 강조 + label) viser 시각화 + world 좌표
```

> **환경 주의 (중요).** mosaic3d(torch 2.2.2)와 unidet3d(torch 2.1.2 + mmdet3d +
> MinkowskiEngine)는 한 env 에 같이 설치할 수 없다. **두 백엔드를 한 서버에서 같이
> 쓰려면 `unidet3d` env 에서 서버를 띄워야 한다** — webapp 의 mosaic3d 경로는 서빙 시
> 미리 만들어둔 `feat.npy` 캐시 위에서 open_clip + DBSCAN 만 돌리므로(spconv/SpUNet101
> 불필요) `unidet3d` env 에서도 잘 돈다. 단 `pip install open_clip_torch viser
> scikit-learn transformers accelerate` 가 그 env 에 추가로 필요하다. 반대로 `mosaic3d`
> env 에서 띄우면 UniDet3D import 가 실패하므로 `Backend` 드롭다운에 `unidet3d` 옵션이
> 아예 안 뜨고(`--enable-unidet3d` 없이) mosaic3d 만 동작한다.

### 1) 환경 셋업 (한 번만)

UniDet3D는 mmdetection3d 1.4.0 + MinkowskiEngine + spconv를 요구해서
기존 `mosaic3d` 환경과는 충돌하기 쉬워. 별도 conda env 권장.
기반 가이드는 [`docs/mmdet_get_started.md`](../../docs/mmdet_get_started.md),
정확한 버전 조합은 [`unidet3d/Dockerfile`](../../unidet3d/Dockerfile)을 따른다.

```bash
# (a) base env — PyTorch 2.1 + CUDA 12.1 (UniDet3D Dockerfile과 동일)
conda create -n unidet3d python=3.10 -y
conda activate unidet3d
pip install torch==2.1.2 torchvision --index-url https://download.pytorch.org/whl/cu121

# (b) OpenMMLab stack — 버전 핀이 중요
pip install --no-deps \
    mmengine==0.9.0 mmdet==3.3.0 mmsegmentation==1.2.0 \
    mmdet3d==1.4.0 mmpretrain==1.2.0

# mmcv는 빌드 필요 (Dockerfile과 동일한 커밋 권장)
pip install -U openmim
mim install "mmcv==2.1.0"   # 안 되면 Dockerfile의 source-build 절차로 fallback

# (c) sparse-conv 백엔드들
pip install spconv-cu120==2.3.6 cumm-cu120==0.5.1
pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
TORCH_CUDA_ARCH_LIST="8.6" pip install --no-deps \
    git+https://github.com/daizhirui/MinkowskiEngine.git \
    --global-option="--blas=openblas" --global-option="--force_cuda"

# (d) 보조 패키지
pip install open3d==0.17.0 plyfile==1.0.2 trimesh==3.21.6 \
            scikit-learn scipy numba==0.57.0 numpy==1.24.1

# (e) webapp_llm용
pip install viser open_clip_torch "transformers>=4.45" accelerate
```

> 8GB 카드에서 OOM 나면 LLM은 다른 env(`--llm-device-map auto`) 또는 작은 모델로.

### 2) 체크포인트 + 데모 데이터 받기

```bash
# pretrained UniDet3D 가중치 (~1GB)
mkdir -p unidet3d/work_dirs
curl -L -o unidet3d/work_dirs/unidet3d.pth \
    https://github.com/filapro/unidet3d/releases/download/v1.0/unidet3d.pth

# 데모 .bin이 없다면 ScanNet++ 씬에서 변환해 unidet3d/data/my_scene.bin 으로 저장
# 포맷: (N, 9) float32 = x, y, z, r, g, b, nx, ny, nz
# 변환 예시는 docs/mmdet_get_started.md의 convert_ply 헬퍼 참고.
```

### 3) 실행

서브모듈이 repo 루트의 `unidet3d/`에 있으면 기본 경로가 자동으로 잡혀서
플래그 없이 `--enable-unidet3d` 만으로 충분 (LLM 모드):

```bash
conda activate unidet3d
# open_clip / viser / sklearn / transformers / accelerate 가 이 env 에 있어야 함
python 3D-segmentation/webapp_llm/server.py \
    --port 8090 \
    --enable-unidet3d \
    --unidet3d-dataset scannetpp \
    --unidet3d-device cuda:0
```

base(비-LLM) 웹앱도 같은 플래그를 받는다:

```bash
conda activate unidet3d
python 3D-segmentation/webapp/server.py \
    --port 8080 --enable-unidet3d --unidet3d-dataset scannetpp --unidet3d-device cuda:0
```

경로를 바꾸고 싶으면 `--unidet3d-root`, `--unidet3d-cfg`, `--unidet3d-ckpt`
로 override 할 수 있다. (`.bin` 입력은 더 이상 쓰지 않는다 — 로드된 cache 씬을 직접 detection.)

웹앱 사용법:

1. 상단 `Backend` 드롭다운에서 `unidet3d` 선택 → UniDet3D 슬라이더(score-thr / top-K /
   show-all / re-run) 가 나타난다.
2. (LLM 웹앱) 자연어 명령 입력 후 **Parse + Localize**, 또는 **UniDet3D: (re)run detection**.
   (base 웹앱) `Query` 에 텍스트 입력 후 **Apply**.
3. 모델 weight 는 최초 1회만 로드되고, 같은 씬에서는 detection 결과를 캐시해 재매칭만 한다.
   LLM이 (LLM 웹앱) 또는 입력 텍스트가 (base 웹앱) → CLIP 으로 bbox 클래스 임베딩과 매칭 →
   Top-K 박스가 빨간 wireframe(★ label)로 강조되고 패널에 world 좌표가 출력된다.

> detector 입력 색상은 `miny-det/convert.py` 와 동일하게 `color/127.5-1` 로 정규화된다.
> 씬에 색이 없으면(gray fallback) detection 품질이 떨어질 수 있다.

## 파일

- [`llm_parser.py`](llm_parser.py) — 로컬 LLM + JSON 추출
- [`server.py`](server.py) — viser UI + LLM → (CLIP 클러스터링 | UniDet3D) 백엔드 글루
- [`../webapp/unidet3d_backend.py`](../webapp/unidet3d_backend.py) — 두 웹앱이 공유하는
  UniDet3D 백엔드(인자/detector/매칭/박스 렌더)
- [`../unidet3d_only/unidet3d_detector.py`](../unidet3d_only/unidet3d_detector.py) —
  UniDet3D wrapper + CLIP class-embed 빌더
