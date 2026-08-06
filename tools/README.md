# tools/ - report tools

이 디렉토리는 드론 순찰/비행 결과를 LLM으로 요약하고, 시뮬레이터가 생성한 결과 파일을 감시해서 자동으로 요약 리포트를 생성하는 도구들을 모아둔 곳입니다.
목적: 시뮬레이터 또는 운영 파이프라인에서 생성된 `traj.json` / `unity_autopilot_3d_result.json` 같은 원시 산출물을 받아

- LLM(Ollama)으로 요약한 표준 리포트(JSON)를 만들고,
- 필요하면 자동으로 서버에 업로드하거나 로컬에 저장해 웹 UI에서 불러볼 수 있게 합니다.

핵심 파일

- report_summarizer.py
    - 역할: 원시 리포트 JSON을 받아 Ollama LLM에 프롬프트를 던져 규격화된 보고서(JSON)를 생성.
    - 동작: Ollama 호출 시 모델 응답을 JSON으로 파싱. 실패(혹은 Ollama 미설치) 시 deterministic한 폴백 요약을 반환.
    - 주의: Ollama 데몬 + 모델이 필요(예: `ollama pull qwen3:1.7b`). 없으면 자동으로 폴백 요약 사용.
- report_watcher.py
    - 역할: 독립 실행형 파일 감시자(데몬) — 지정 디렉터리를 감시하고 새로 생성된 `traj.json` 또는 `unity_autopilot_3d_result.json` 파일을 찾아 서버의 `POST /api/report`에 전송.
    - 사용 환경: 운영/테스트 환경에서 web_server가 별도로 실행 중일 때 권장(프로세스 분리).
    
- report_watcher_thread.py
    - 역할: web_server 프로세스 내부에서 쓰기 위한 스레드형 감시기 — 감지 즉시 `report_summarizer.summarize()`를 직접 호출하거나, 서버의 `/api/report`로 POST 가능.
    - 사용 환경: web_server 내부에서 파일 감시 + 요약을 직접 수행하고 싶을 때 사용.

자동화의 두가지 방식 (선택 1)

- (A) 시뮬레이터가 traj.json 생성 → **report_watcher.py(독립형)** 가 파일 감지 → POST /api/report → 웹서버가 tools/report_summarizer.summarize() 호출(또는 서버 자체에서 요약) → web/uploads/reports/<id>.json 저장
- (B) 혹은 웹서버 내부 (**report_watcher_thread.py** )watcher_thread가 직접 파일을 읽어 report_summarizer.summarize() 호출 → 로컬에 .report.json 저장 또는 서버 API로 전송

Required Python packages (example):
- watchdog
- requests
- ollama (if you want Ollama client)
Install:
  pip install watchdog requests ollama

Recommended workflow:
1. Add tools/ files to a new branch.
2. On your target machine install dependencies.
3. Run Ollama and pull the model (e.g. `ollama pull qwen3:1.7b`).
4. Start web server (or keep it stopped if you will call summarizer locally).
5. Start watcher:
   - Standalone watcher:
       python tools/report_watcher.py --watch-dir ./simulator/output --server http://localhost:8000
   - Thread watcher (inside web_server):
       from tools.report_watcher_thread import ReportWatcherThread
       from tools import report_summarizer
       th = ReportWatcherThread(['simulator/output'], post_url=None, call_local=True, summarizer_callable=report_summarizer.summarize)
       th.start()
