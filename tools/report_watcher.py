# tools/report_watcher.py 독립 실행형 watcher
"""
Standalone watcher that monitors one or more directories for new traj.json or
unity_autopilot_3d_result.json files and POSTs them to the server's /api/report.

Dependencies:
- pip install watchdog requests

Usage:
  python tools/report_watcher.py --watch-dir ./simulator/output --server http://localhost:8000
"""
import argparse
import json
import time
import os
import threading
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

TARGET_NAMES = {'traj.json', 'unity_autopilot_3d_result.json'}

class _NewFileHandler(FileSystemEventHandler):
    def __init__(self, server_url, watch_dir, delay=1.0):
        self.server_url = server_url.rstrip('/')
        self.watch_dir = watch_dir
        self.delay = delay
        super().__init__()

    def on_created(self, event):
        if event.is_directory:
            return
        path = os.path.basename(event.src_path)
        if path in TARGET_NAMES:
            threading.Thread(target=self._handle_file, args=(event.src_path,)).start()

    def _handle_file(self, path):
        # Wait briefly for file to be fully written
        time.sleep(self.delay)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
        except Exception as e:
            print(f"[watcher] Failed to read {path}: {e}")
            return

        try:
            url = f"{self.server_url}/api/report"
            resp = requests.post(url, json=raw, timeout=30)
            if resp.status_code == 200:
                print(f"[watcher] Posted {path} -> {url}, response: {resp.json().get('id')}")
            else:
                print(f"[watcher] POST {url} returned {resp.status_code}: {resp.text}")
        except Exception as e:
            print(f"[watcher] HTTP POST failed: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--watch-dir', '-w', required=True, help='Directory to watch (recursive)')
    ap.add_argument('--server', '-s', default='http://127.0.0.1:8000', help='Web server base URL')
    ap.add_argument('--delay', type=float, default=1.0, help='Seconds to wait after file creation before reading')
    args = ap.parse_args()

    event_handler = _NewFileHandler(args.server, args.watch_dir, delay=args.delay)
    observer = Observer()
    observer.schedule(event_handler, args.watch_dir, recursive=True)
    observer.start()
    print(f"[watcher] Watching {args.watch_dir} -> {args.server}/api/report")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    main()