"""
UDP-over-TCP relay for network-separated setups (laptop behind Wi-Fi NAT).

The pipeline needs server -> laptop UDP 9000, but campus Wi-Fi usually blocks
inbound to the laptop while the laptop CAN reach the server. This relay
inverts the transport: the laptop connects OUT to the server over one TCP
connection, and all Tello UDP traffic (commands, "ok" replies, state stream)
is framed over it as JSON lines. Stdlib only — copy this single file to the
laptop and run it with any Python 3.9+.

Server (GPU box), instead of --unity-host <laptop-ip>:
    python simulator/bridge/relay.py server            # TCP :9010, UDP :9000
    python 3D-segmentation/webapp_llm_v2/server.py ... --sim --unity-host 127.0.0.1

Laptop (Unity running, Play mode):
    python relay.py client --server-host 166.104.223.32

Optional shared secret on both sides: --token <string>.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time

BUF = 4096


def _send_frame(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))


def _read_frames(sock: socket.socket):
    """Yield newline-delimited JSON frames until the connection dies."""
    buf = b""
    while True:
        chunk = sock.recv(BUF)
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                try:
                    yield json.loads(line.decode("utf-8"))
                except json.JSONDecodeError:
                    pass


# --------------------------------------------------------------------- server
def run_server(tcp_port: int, cmd_port: int, state_port: int, token: str) -> None:
    """Runs on the GPU box. Pretends to be Unity on localhost:<cmd_port>."""
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", cmd_port))

    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp.bind(("0.0.0.0", tcp_port))
    tcp.listen(1)

    lock = threading.Lock()
    client: socket.socket | None = None
    bridge_addr: tuple | None = None  # UDP source of the pipeline's bridge

    def udp_loop() -> None:
        nonlocal bridge_addr
        while True:
            data, addr = udp.recvfrom(BUF)
            bridge_addr = addr
            with lock:
                c = client
            if c is None:
                continue  # no laptop connected; command dropped (bridge times out)
            try:
                _send_frame(c, {"ch": "cmd", "data": data.decode("utf-8", "replace")})
            except OSError:
                pass

    threading.Thread(target=udp_loop, daemon=True).start()
    print(f"[relay-server] UDP command endpoint 127.0.0.1:{cmd_port} "
          f"(run the pipeline with --unity-host 127.0.0.1)")
    print(f"[relay-server] waiting for laptop on TCP :{tcp_port} ...")

    while True:
        conn, remote = tcp.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        frames = _read_frames(conn)
        try:
            hello = next(frames, None)
        except OSError:
            hello = None
        if hello is None or hello.get("ch") != "hello" or hello.get("token", "") != token:
            print(f"[relay-server] rejected {remote[0]} (bad hello/token)")
            conn.close()
            continue
        with lock:
            if client is not None:
                try:
                    client.close()
                except OSError:
                    pass
            client = conn
        print(f"[relay-server] laptop connected: {remote[0]}")
        try:
            for frame in frames:
                ch = frame.get("ch")
                data = frame.get("data", "").encode("utf-8")
                if ch == "reply" and bridge_addr is not None:
                    udp.sendto(data, bridge_addr)
                elif ch == "state":
                    udp.sendto(data, ("127.0.0.1", state_port))
        except OSError:
            pass
        print(f"[relay-server] laptop disconnected: {remote[0]}")
        with lock:
            if client is conn:
                client = None


# --------------------------------------------------------------------- client
def run_client(server_host: str, tcp_port: int, unity_host: str,
               unity_port: int, state_listen_port: int, token: str) -> None:
    """Runs on the laptop next to Unity. Reconnects forever."""
    while True:
        try:
            _client_session(server_host, tcp_port, unity_host, unity_port,
                            state_listen_port, token)
        except (OSError, ConnectionError) as e:
            print(f"[relay-client] connection lost ({e}); retrying in 3s ...")
        time.sleep(3.0)


def _client_session(server_host: str, tcp_port: int, unity_host: str,
                    unity_port: int, state_listen_port: int, token: str) -> None:
    tcp = socket.create_connection((server_host, tcp_port), timeout=10.0)
    tcp.settimeout(None)
    tcp.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    _send_frame(tcp, {"ch": "hello", "token": token})
    print(f"[relay-client] connected to {server_host}:{tcp_port}")

    # Commands go to Unity from this socket; Unity's "ok" replies come back to
    # it, and Unity records its address, so the state stream targets this host.
    cmd_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_udp.settimeout(0.5)

    state_udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_udp.bind(("0.0.0.0", state_listen_port))
    state_udp.settimeout(0.5)

    dead = threading.Event()

    def pump_udp_to_tcp(sock: socket.socket, ch: str) -> None:
        while not dead.is_set():
            try:
                data, _ = sock.recvfrom(BUF)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                _send_frame(tcp, {"ch": ch, "data": data.decode("utf-8", "replace")})
            except OSError:
                dead.set()
                break

    threading.Thread(target=pump_udp_to_tcp, args=(cmd_udp, "reply"), daemon=True).start()
    threading.Thread(target=pump_udp_to_tcp, args=(state_udp, "state"), daemon=True).start()

    try:
        for frame in _read_frames(tcp):
            if frame.get("ch") == "cmd":
                cmd_udp.sendto(frame.get("data", "").encode("utf-8"),
                               (unity_host, unity_port))
    finally:
        dead.set()
        for s in (cmd_udp, state_udp, tcp):
            try:
                s.close()
            except OSError:
                pass
        raise ConnectionError("server closed the relay connection")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("server", help="run on the GPU box")
    s.add_argument("--tcp-port", type=int, default=9010)
    s.add_argument("--cmd-port", type=int, default=9000,
                   help="local UDP port the pipeline's bridge sends commands to")
    s.add_argument("--state-port", type=int, default=9002,
                   help="local UDP port the state stream is forwarded to")
    s.add_argument("--token", default="")

    c = sub.add_parser("client", help="run on the laptop next to Unity")
    c.add_argument("--server-host", required=True)
    c.add_argument("--tcp-port", type=int, default=9010)
    c.add_argument("--unity-host", default="127.0.0.1")
    c.add_argument("--unity-port", type=int, default=9000)
    c.add_argument("--state-listen-port", type=int, default=9002,
                   help="port Unity streams state to (TelloSimulator statePort)")
    c.add_argument("--token", default="")

    args = ap.parse_args()
    if args.mode == "server":
        run_server(args.tcp_port, args.cmd_port, args.state_port, args.token)
    else:
        run_client(args.server_host, args.tcp_port, args.unity_host,
                   args.unity_port, args.state_listen_port, args.token)


if __name__ == "__main__":
    main()
