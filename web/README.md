# web — 흉가 드론 관제 (HAUNTED OPS) 웹 UI

디자인 시안 **"실내 드론 목적지 선택 UI"** 를 그대로 가져온 관제 화면에,
**두 점(출발/도착)을 찍으면 학습된 강화학습(SAC) 정책이 실제 경로를 만들어 주는**
백엔드를 붙인 것.

`playground/` 는 실험용 프로토타입 모음이고, **실제로 돌려서 쓰는 웹 앱은 이 폴더**다.
서버는 최상위 `web_server.py`.

## 실행

```bash
conda activate tello
python web_server.py            # 저장소 최상위에서. http://localhost:8000
```

정책 로드에 약 6초가 걸리고(1회), 이후 경로 계획 한 번은 0.05~0.5초다.
`--no-rl` 로 띄우면 정책 없이 UI 만 서빙되고, 프론트는 방그래프 직선 경로로 폴백한다.

## 파일

| 파일 | 역할 |
|---|---|
| `HAUNTED OPS.dc.html` | 게임 셸 — 로비 / 관제(지도) / 브리핑 / Unity 피드 / 작전 / 파지 / 성공 / 게임오버 / 기록 / 설정 |
| `드론 관제.dc.html` | 지도 화면. `dc-import` 로 위 셸의 '관제' 단계에 삽입된다. **2점 선택 + RL 경로**가 여기 있다 |
| `support.js` | 시안이 쓰는 템플릿 런타임(dc-runtime). React/ReactDOM UMD 를 unpkg 에서 자동 로드하고 `<x-dc>` 템플릿을 렌더한다 — 최초 실행 시 인터넷 필요 |
| `uploads/` | `web_meta.json`(방/문/층 메타), `floor_*.png`(층별 평면도), `detections.json`(가구 인스턴스) |
| `index.html` | `HAUNTED OPS.dc.html` 로 보내는 리다이렉트 |

`uploads/` 의 평면도·메타는 `playground/demo_house/export_web_assets.py` 산출물이고,
`detections.json` 만 디자인 프로젝트에서 함께 가져왔다. 방을 다시 찍거나 점군이
바뀌어 에셋을 갱신할 땐 이 폴더로 직접 내보내면 된다:

```bash
python playground/demo_house/export_web_assets.py --out web/uploads
```

(인자 없이 실행하면 기존 `playground/demo_house/web_assets/` 로 나가고, 그건 구
프로토타입 `playground/demo_house/web_ui/` 가 쓰는 사본이다.)

## 2점 선택 → 강화학습 경로

지도에서 **첫 클릭 = 출발점, 두 번째 클릭 = 도착점**. 두 점이 모이면 곧바로
`POST /plan` 을 쳐서 서버가 정책을 굴린 결과를 받아 온다.

```
POST /plan   {"start":[x,y,z], "goal":[x,y,z], "people":0, "shield":false}
     →       {"engine":"SAC · model_geo_best", "success":true, "steps":83,
              "bumps":0, "dist":0.50, "flown":27.89, "ms":537,
              "start":[...], "goal":[...],        // 실제로 쓰인(스냅된) 좌표
              "path":[[x,y,z], ...]}              // 3D 궤적
```

- 서버는 `playground/reinforce_learning/model_geo_best.zip` 정책을 **결정론**으로
  롤아웃한다. 방그래프 BFS 나 RRT* 가 아니라 정책 단독 추론이 경로를 만든다.
- 벽 위나 집 밖을 찍어도 환경이 가장 가까운 도달 가능 셀로 스냅한다
  (응답의 `start`/`goal` 이 스냅 후 좌표, `requested_*` 가 클릭 원본).
- 궤적의 층은 z 로 판정한다(`z<0.2` → 1층, `z<2.9` → 2층, 그 외 3층).
  계단으로 층을 넘는 구간은 두 층의 평면도에 나눠 그려진다.
- 서버가 없거나 실패하면 프론트가 **방그래프 직선 경로로 자동 폴백**하고
  사이드바 'RL 경로' 패널에 실패 사유를 띄운다.

「비행 시작」을 누르면 이 궤적이 그대로 브리핑 화면의 3층 평면도와,
Unity 화면의 **「관제 지도에서 RL 경로 재생」** 버튼(작전 화면 궤적 재생)으로 넘어간다.

## 주의

- `web_server.py` 의 `ENV_KW` 는 학습 당시 설정이다. 바꾸면 정책 성능이 무너진다.
  특히 `clearance=0.12` 는 반드시 명시(기본 0.235로 env 를 만들면 `geo_graph_cache`
  가 덮여서 다음 실행이 캐시를 1분간 재생성한다).
- `support.js` 는 생성된 런타임이라 직접 수정하지 말 것.
- 환경 객체는 재진입 불가라 서버가 Lock 으로 요청을 직렬화한다(동시 요청은 순차 처리).
