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

## 파일

- [`llm_parser.py`](llm_parser.py) — 로컬 LLM + JSON 추출
- [`server.py`](server.py) — viser UI + LLM → CLIP → 클러스터링 글루
