# tools/ - report tools

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