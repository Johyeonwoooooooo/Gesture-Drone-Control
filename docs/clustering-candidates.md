# 3D 히트맵 클러스터링 기반 후보 추출

## 1. 배경

[project-overview.md](project-overview.md)의 **3. 3D Segmentation** 단계는
사용자 자연어 질의(예: `"a tv"`)에 대해 씬의 모든 점에 대한 cosine 유사도 히트맵을
생성한다 (Mosaic3D SpUNet101 + CLIP 텍스트 헤드).

기존 webapp(`3D-segmentation/webapp/server.py`)은 이 히트맵을 그대로 컬러로 입혀서
시각화만 하기 때문에, 경로 계획기에 넘길 **단일 좌표(들)** 가 바로 나오지 않는다.
같은 라벨에 해당하는 점이 씬 안에 여러 군데 존재할 수 있으므로
(예: "의자"가 거실에 4개, 식탁에 6개), 단일 argmax 점을 잡으면 잘못된 인스턴스를
선택할 위험이 있다.

이 문서는 히트맵을 **공간적으로 클러스터링**하여 여러 후보를 추출하고
각 후보를 점수로 랭킹하는 모듈을 설명한다. 출력은 `(x, y, z)` 좌표를 가진 후보
리스트이며 LLM이나 path planner가 다음 단계에서 골라 쓸 수 있다.

---

## 2. 파이프라인

```
cache/feat/<region>/feat.npy   (N, 768) fp16  ─┐
cache/feat/<region>/coord.npy  (N, 3)  float32 ┘
                                       │
                       자연어 query ───┤
                                       ▼
                        [1] CLIP 텍스트 인코딩
                                       │
                                       ▼
                        [2] cosine sim → 점별 score
                                       │
                                       ▼
                        [3] 상위 percentile / threshold 컷
                                       │
                                       ▼
                        [4] DBSCAN 공간 클러스터링
                                       │
                                       ▼
                        [5] 후보별 집계 (centroid, bbox, score)
                                       │
                                       ▼
                        [6] mean_score · √n_points 로 랭킹
                                       │
                                       ▼
                        Top-K candidates  (JSON)
```

---

## 3. 입력 데이터

[3D-segmentation/inference/run_inference.py](../3D-segmentation/inference/run_inference.py)
가 만든 캐시를 그대로 사용한다.

| 파일 | shape | dtype | 의미 |
|---|---|---|---|
| `cache/feat/<region>/feat.npy` | (N, 768) | fp16 | 점별 CLIP-text-aligned feature |
| `cache/feat/<region>/coord.npy` | (N, 3) | float32 | 점별 **원본 월드 좌표** (m 단위) |

> `coord.npy`는 webapp이 카메라 정렬용으로 빼는 bbox-center를 적용하기 **전** 상태다.
> 즉 클러스터링 결과로 나오는 좌표는 그대로 path planner의 월드 좌표계와 호환된다.

Region 명은 `cache/feat/`에서 직접 확인할 수 있다 (`ls 3D-segmentation/cache/feat/`).

---

## 4. 알고리즘 선택 이유

### 4.1 Threshold: top-percentile (default)

CLIP 점수의 절대 값은 텍스트 프롬프트와 씬에 따라 분포가 크게 바뀐다 (예: 흔한 단어는
전체적으로 높게, 희귀한 단어는 전체적으로 낮게). 절대 임계값 대신 **상위 X%**
컷을 기본으로 쓰면 새로운 query마다 임계값을 손볼 필요가 없다.
필요시 `--threshold`로 절대값 컷을 강제할 수 있다 (webapp의 `Threshold` 슬라이더와
동일한 의미).

### 4.2 Clustering: DBSCAN

- **eps**: 동일 인스턴스에 속한 점들의 최대 이웃 거리. 추론 grid_size = 0.02m이고
  실내 가구 크기는 보통 0.5~2m이므로 **0.25m**가 기본값으로 적합 (≈ 12.5 voxel).
- **min_points**: 클러스터로 인정할 최소 점 수. 너무 작은 클러스터는 노이즈 가능성이
  크므로 **40**을 기본값으로 둔다 (region 한 채당 200k 점, 상위 5% = 10k 점 기준
  0.4% 미만은 버린다).
- 클러스터 개수를 미리 모르므로 K-means 같은 비-밀도 기반 알고리즘은 부적합.
  HDBSCAN도 가능하지만 추가 deps가 필요하고, 우리 데이터는 grid_size가 균일해서
  DBSCAN으로 충분하다.

### 4.3 Ranking: `mean_score · √n_points`

- 단순 mean_score만 쓰면 **표면 한 점**만 강하게 반응한 작은 클러스터가 1위가 되기 쉽다.
- n_points만 쓰면 **벽/바닥** 같은 거대 surface에 우연히 비슷한 점이 깔린 케이스가 1위가 된다.
- `√n_points`로 가중하면 "어느 정도 큰 + 평균 점수가 높은" 후보를 안정적으로 끌어올린다.

---

## 5. 사용 방법

### 5.0 웹앱 (Viser) 시각화

웹앱(`3D-segmentation/webapp/server.py`)에 **`cluster` 모드**가 추가되어 있어,
같은 GUI에서 후보를 시각화할 수 있다.

1. **Mode** 드롭다운 → `cluster` 선택
2. **Query** 칸에 자연어 프롬프트 입력 (예: `a tv`)
3. 우측 슬라이더로 파라미터 조절
   - `Cluster: top-percentile` — 활성 점 컷 (default 95)
   - `Cluster: eps (m)` — DBSCAN 이웃 반경 (default 0.25)
   - `Cluster: top-K` — 표시할 후보 개수 (default 5)
   - `Cluster: marker radius (m)` — sphere 마커 크기 (default 0.15)
4. **Apply** 클릭

렌더링 결과:
- 점구름 위에 **단일 쿼리 모드와 동일한 히트맵**이 깔린다 (top-percentile 컷 기준).
- 각 후보 centroid에 **rank별 색상의 sphere**가 표시된다 (Set1 팔레트, rank 0=빨강).
- 후보의 **bounding box가 같은 색의 wireframe**으로 그려진다.
- 각 후보 위에 `#rank n=점수 s=score` 라벨이 떠 있다.
- 좌측 패널에 후보별 좌표/점수/크기가 정렬된 리스트로 노출된다.

> region을 바꾸거나 `cluster` → 다른 모드로 전환하면 마커는 자동으로 제거된다.
> 슬라이더(eps, top-percentile 등)를 바꾼 뒤에는 **Apply**를 다시 눌러야 반영된다.

### 5.1 CLI

```bash
source /data1/workspaces/jgshin22/miniconda3/etc/profile.d/conda.sh
conda activate mosaic3d
cd /data1/workspaces/jgshin22/Gesture-Drone-Control/3D-segmentation

python inference/cluster_candidates.py \
    --region 17DRP5sb8fy_00 \
    --query "a tv" \
    --top-percentile 95 \
    --eps 0.25 \
    --min-points 40 \
    --top-k 5 \
    --device cuda:1
```

stdout에 JSON이 그대로 찍힌다. 파일로 저장하려면 `--out candidates.json` 추가.

### 5.2 Python module

```python
from pathlib import Path
from inference.cluster_candidates import extract_candidates, ClusterParams

result = extract_candidates(
    region="17DRP5sb8fy_00",
    query="a tv",
    feat_dir=Path("3D-segmentation/cache/feat"),
    params=ClusterParams(top_percentile=95, eps=0.25, min_points=40, top_k=5),
    device="cuda:1",
)

for c in result["candidates"]:
    print(f"[{c['rank']}] center={c['center']} "
          f"n={c['n_points']} mean={c['mean_score']:.4f}")
```

---

## 6. 출력 스키마

```jsonc
{
  "region": "17DRP5sb8fy_00",
  "query": "a chair",
  "params": {
    "top_percentile": 95.0,
    "threshold": null,
    "eps": 0.25,
    "min_points": 40,
    "top_k": 5
  },
  "stats": {
    "n_total_points": 201726,
    "score_min": -0.0665,
    "score_max":  0.0174,
    "score_p95":  -0.0034,
    "n_active_points": 10089,   // top-percentile 컷을 통과한 점 수
    "n_clusters": 3,            // DBSCAN이 만든 유효 클러스터 개수 (noise 제외)
    "n_noise_points": 84,
    "elapsed_s": 0.341
  },
  "candidates": [
    {
      "rank": 0,
      "cluster_id": 1,
      "center":   [-10.90, 1.51, 0.45],   // 월드 좌표 (m)
      "bbox_min": [-11.16, 1.03, -0.02],
      "bbox_max": [-10.37, 1.99,  1.34],
      "n_points": 3261,
      "mean_score": 0.00153,
      "max_score":  0.01585
    },
    // ... rank 1, 2, ...
  ]
}
```

> **score 절대값에 의미 부여하지 말 것.** 본 모델 + CLIP 헤드의 cosine sim은
> 분포가 좁고 음수도 많이 나온다 (webapp의 기본 슬라이더 범위가 `[-0.05, 0.5]`인 것도
> 같은 이유). 비교는 **같은 query 안에서의 상대 랭킹**으로만 한다.

---

## 7. 파라미터 튜닝 가이드

| 상황 | 조정 |
|---|---|
| 후보가 0개 | `--top-percentile 90` 으로 풀기, 또는 `--min-points 20` 으로 낮추기 |
| 후보가 너무 많고 잘게 쪼개짐 | `--eps 0.4` ~ `0.5`로 키워서 인접 인스턴스 합치기 |
| 큰 surface(벽/바닥)가 1위에 자꾸 잡힘 | query를 더 specific하게 (`"a tv on the wall"`), 또는 `--top-percentile 98`로 더 strict |
| 작은 객체(리모컨 등) | `--min-points 10`, `--eps 0.1` 로 작게 |

`stats.n_active_points`, `n_clusters`, `n_noise_points`를 보고 한 번 더 호출하면서
파라미터를 좁히는 워크플로를 권장.

---

## 8. 한계 및 향후 개선

1. **방(room)/floor 정보 미사용** — 현재는 region 한 채 안에서만 클러스터링한다.
   "옆 방 TV" 같은 query는 LLM이 region을 먼저 좁혀 주거나, multi-region 후보를
   동시에 비교하는 상위 레이어가 필요하다.
2. **객체-vs-vs-surface 구분 없음** — 클러스터의 형상 (얇은 평면 vs 박스형)을 따로
   판별하지 않는다. bbox 종횡비 휴리스틱으로 surface 후보를 페널티화하는 것이
   가벼운 개선 방안.
3. **점수 정규화** — 현재 raw cosine sim으로 랭킹한다. negative-prompt 대비 정규화
   (예: `"a TV"` 점수에서 `"surface"` 점수를 빼는 식) 를 더하면 false positive를
   더 거를 수 있다.
4. **드론 접근 가능성 미고려** — 후보 좌표 자체는 객체 표면 점들의 centroid라
   드론이 직접 그 좌표로 가면 가구 안으로 들어간다. path planner 측에서 후보의
   `bbox`를 받아 안전한 standoff 거리를 두고 viewpoint를 찾도록 해야 한다.

---

## 9. 관련 파일

- 구현: [3D-segmentation/inference/cluster_candidates.py](../3D-segmentation/inference/cluster_candidates.py)
- 입력 캐시 생성: [3D-segmentation/inference/run_inference.py](../3D-segmentation/inference/run_inference.py)
- 시각화 (참고): [3D-segmentation/webapp/server.py](../3D-segmentation/webapp/server.py)
- 프로젝트 컨텍스트: [project-overview.md](project-overview.md)
