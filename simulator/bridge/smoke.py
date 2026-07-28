"""
Network smoke test for the Unity simulator link. Run BEFORE the full server to
isolate firewall/routing problems:

  python simulator/bridge/smoke.py --unity-host <laptop-ip> [--fly]

Checks: `command` -> "ok" (UDP 9000 reachable), state packet arrives on 9002
(reverse path reachable), and with --fly a short takeoff/hover/land.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unity_bridge import UnityTelloBridge  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unity-host", required=True)
    ap.add_argument("--unity-port", type=int, default=9000)
    ap.add_argument("--local-port", type=int, default=9001)
    ap.add_argument("--state-port", type=int, default=9002)
    ap.add_argument("--fly", action="store_true", help="takeoff, hover 2s, land")
    args = ap.parse_args()

    bridge = UnityTelloBridge(args.unity_host, args.unity_port, args.local_port, args.state_port)
    bridge.connect()
    try:
        reply = bridge.initialize_sdk()
        if reply == "timeout":
            print(f"FAIL: no reply from {args.unity_host}:{args.unity_port} — "
                  f"Unity not in Play mode, wrong IP, or inbound UDP {args.unity_port} blocked.")
            return 1
        print(f"OK: command -> {reply!r}")

        state = bridge.wait_for_state(timeout=3.0)
        if state is None:
            print(f"FAIL: no state packet on UDP {args.state_port} — "
                  f"inbound UDP {args.state_port} blocked on this machine?")
            return 1
        print(f"OK: state pos=({state.x:.2f},{state.y:.2f},{state.z:.2f}) "
              f"flying={state.flying} t={state.time:.1f}")

        bridge.send_status("smoke test OK")

        if args.fly:
            print("takeoff...")
            bridge.takeoff()
            time.sleep(2.0)
            s = bridge.get_latest_state()
            print(f"hover at y={s.y:.2f}" if s else "hover (no state?)")
            bridge.land()
            print("landed")
        print("SMOKE TEST PASSED")
        return 0
    finally:
        bridge.close()


if __name__ == "__main__":
    raise SystemExit(main())
