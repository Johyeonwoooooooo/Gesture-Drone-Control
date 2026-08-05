# tools/report_watcher_thread.py
"""
Threaded watcher that can be started from web_server (or any process) to watch
directories and either call the local summarizer directly or POST to a server endpoint.

Usage (import and start):
    from tools.report_watcher_thread import ReportWatcherThread
    th = ReportWatcherThread(watch_dirs=['simulator/output'], post_url=None, call_local=True)
    th.start()
    # to stop: th.stop()
"""
import threading
import time
import os
import json
from typing import List, Optional, Callable

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    Observer = None  # watchdog not installed

class _Handler(FileSystemEventHandler):
    def __init__(self, callback, delay=1.0):
        self.callback = callback
        self.delay = delay
        super().__init__()

    def on_created(self, event):
        if event.is_directory:
            return
        # only react to filenames commonly emitted by simulator
        name = os.path.basename(event.src_path)
        if name in ('traj.json', 'unity_autopilot_3d_result.json'):
            # call callback in separate thread to avoid blocking watchdog
            threading.Thread(target=self._delayed_cb, args=(event.src_path,)).start()

    def _delayed_cb(self, path):
        time.sleep(self.delay)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        except Exception:
            return
        try:
            self.callback(raw, path)
        except Exception:
            return

class ReportWatcherThread:
    def __init__(self, watch_dirs: List[str], post_url: Optional[str] = None,
                 call_local: bool = True, summarizer_callable: Optional[Callable] = None,
                 delay: float = 1.0):
        """
        - watch_dirs: list of directories to watch (non-recursive recommended)
        - post_url: if set, POST raw payloads to this URL (/api/report)
        - call_local: if True and summarizer_callable is provided, call it instead of POST
        - summarizer_callable: function(raw_dict) -> report_dict
        """
        self.watch_dirs = watch_dirs
        self.post_url = post_url
        self.call_local = call_local
        self.summarizer_callable = summarizer_callable
        self.delay = delay
        self._observer = None
        self._running = False

    def start(self):
        if Observer is None:
            raise RuntimeError("watchdog not installed")
        if self._running:
            return
        self._running = True
        self._observer = Observer()
        for d in self.watch_dirs:
            handler = _Handler(self._dispatch, delay=self.delay)
            self._observer.schedule(handler, d, recursive=False)
        self._observer.start()

    def _dispatch(self, raw: dict, path: str):
        # When a file is detected, either call local summarizer or POST to server
        if self.call_local and self.summarizer_callable:
            try:
                report = self.summarizer_callable(raw)
                # If summarizer returns a dict, save locally next to file (optional)
                out_path = os.path.splitext(path)[0] + '.report.json'
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                print(f"[watcher-thread] Summarized and saved {out_path}")
            except Exception as e:
                print(f"[watcher-thread] Summarizer error: {e}")
        elif self.post_url:
            import requests
            try:
                resp = requests.post(self.post_url.rstrip('/') + '/api/report', json=raw, timeout=30)
                print(f"[watcher-thread] POST {self.post_url} -> {resp.status_code}")
            except Exception as e:
                print(f"[watcher-thread] POST error: {e}")

    def stop(self):
        if not self._running or self._observer is None:
            return
        self._observer.stop()
        self._observer.join()
        self._running = False