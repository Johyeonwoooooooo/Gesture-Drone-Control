# miny-det — standalone UniDet3D 탐지 + CLIP 쿼리 뷰어

`.npy` 점군 → UniDet3D 3D bbox 탐지 → 박스별 CLIP 임베딩 → viser 에서 자연어로
박스 강조. 4개 스크립트를 **순서대로** 돌리는 오프라인 파이프라인.
(`3D-segmentation/` 웹앱의 UniDet3D 백엔드가 여기서 리팩터된 원본 — 이 폴더는 참조용으로 유지.)

## 파이프라인

```
coord/color/normal.npy
        │  ① convert.py   (voxel 0.01 다운샘플)
        ▼
  (N,9) .bin  = xyz,rgb,normal
        │  ② infer.py     (UniDet3D, 80k 점 random subsample)
        ▼
  det_*.pkl   = bboxes,labels,scores,classes,points,box_pts
        │  ③ clip_index.py (박스 class → CLIP ViT-B/32 텍스트 임베딩)
        ▼
  det_*_clip_index.pkl
        │  ④ app.py        (viser :8081, 텍스트 쿼리 → top-K 박스 강조)
        ▼
  http://<host>:8081
```

> **OOM 안 나는 이유**: ① `convert.py` 가 voxel 0.01 로 한 번, ② `infer.py` 가
> `random_sample_points(max_points=80000)` 로 또 한 번 점을 줄여서 UniDet3D 에 넣는다.
> 웹앱 UniDet3D 백엔드는 cache 씬 전체 점군을 그대로 넣어서 OOM 이 난다(별개 이슈).

## 환경

- **①②**: `unidet3d` env (torch 2.1.2 + mmdet3d 1.4.0 + MinkowskiEngine + spconv).
  셋업은 [`../3D-segmentation/webapp_llm/README.md`](../3D-segmentation/webapp_llm/README.md) §1.
- **③④**: `clip`(OpenAI CLIP, `pip install git+https://github.com/openai/CLIP.git`) +
  `viser` + torch. (open_clip 아님 — ③④는 OpenAI CLIP `ViT-B/32` 사용.)
  `unidet3d` env 에 `pip install ftfy regex viser git+https://github.com/openai/CLIP.git`
  해두면 한 env 에서 ①~④ 다 돌릴 수 있다.

```bash
source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh
conda activate unidet3d
```

## 실행 (한 번씩 순서대로)

스크립트 상단 상수(경로/CFG/CKPT/INPUT/OUT)를 환경에 맞게 먼저 확인/수정.
체크포인트는 `unidet3d/work_dirs/unidet3d.pth` 에 있어야 한다.

```bash
cd /home/jgshin22/work/Gesture-Drone-Control/miny-det

python convert.py       # ① coord/color/normal.npy → data/<scene>.bin
python infer.py         # ② <scene>.bin → data/det_<scene>.pkl
python clip_index.py    # ③ det_<scene>.pkl → det_<scene>_clip_index.pkl
python app.py           # ④ viser 서버 (port 8081)
```

브라우저에서 `http://<host>:8081` 접속 → `Query` 폴더에 텍스트 입력
(예: `chair`, `toilet`) → top-K(기본 3) 박스가 빨강으로 강조, 박스 안 점은 노랑.

## 파일

| 파일 | 입력 → 출력 | env |
|---|---|---|
| `convert.py` | `coord/color/normal.npy` → `data/<scene>.bin` (N,9) | unidet3d |
| `infer.py` | `<scene>.bin` → `data/det_<scene>.pkl` | unidet3d |
| `clip_index.py` | `det_<scene>.pkl` → `det_<scene>_clip_index.pkl` | clip |
| `app.py` | `det_<scene>_clip_index.pkl` → viser :8081 | clip + viser |

> 상수는 각 스크립트 맨 위에 하드코딩(`INDEX_FILE`, `CFG`, `CKPT`, `INPUT`, `OUT`,
> `DATASET_NAME='scannetpp'` 등). 다른 씬으로 바꾸려면 거기만 수정.
