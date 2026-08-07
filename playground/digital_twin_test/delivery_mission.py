# -*- coding: utf-8 -*-
"""
delivery_mission.py - 자율 화물 배달 미션 데모

Unity 시뮬레이터(tello_simulator, Play 상태)와 UDP로 통신하면서
씬에 배치된 화물(CarryableProp)을 전부 배달 구역(DropZone)으로 운반한다.

흐름:
  1. 이륙 (takeoff)
  2. "objects" 명령으로 화물/배달구역 위치 수신 + 유령(동적 장애물) 임포트
  3. 각 화물에 대해: 화물 위로 비행 → grab → 배달구역으로 비행 → drop
     (이동 구간에서는 상태 스트림의 유령 위치를 받아 퍼텐셜 필드로 우회)
  4. 전부 배달되면 착륙 후 결과 리포트 출력

자율 회피:
  Unity가 20Hz 상태 스트림에 유령 위치("ghosts")를 실어 보내고, goto()가 목표
  인력 + 유령 반발을 합쳐 rc를 계산한다. 화물 접근/투하 등 정밀 하강 구간에서는
  avoid=False로 회피를 꺼서 정렬을 유지한다.

실행 전 준비:
  - Unity에서 tello_simulator 프로젝트를 열고 Play 버튼을 누른다.
  - (오브젝트 배치는 SimPropManager가 자동으로 수행한다)

사용법:
  python delivery_mission.py
"""

import json
import math
import socket
import time

UNITY_IP = "127.0.0.1"
CMD_PORT = 9000     # 유니티 명령 수신 포트
STATE_PORT = 9002   # 유니티가 상태 JSON을 쏘는 포트


class SimClient:
    """Unity 시뮬레이터와의 UDP 통신 (명령 + 상태 스트림)"""

    def __init__(self):
        # 명령용 소켓 (임시 포트 — 유니티는 이 주소를 기억해 상태를 보낸다)
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.cmd_sock.bind(("0.0.0.0", 0))
        self.cmd_sock.settimeout(0.3)
        # 상태 수신용 소켓 (유니티 → 127.0.0.1:9002, 20Hz)
        self.state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_sock.bind(("0.0.0.0", STATE_PORT))
        self.state_sock.settimeout(0.05)

        self.state = None     # 마지막 드론 상태
        self.objects = None   # 마지막 objects 응답

    def send(self, cmd: str) -> str:
        """명령 전송 후 ack 수신 (유니티가 꺼져도 크래시하지 않음)"""
        try:
            self.cmd_sock.sendto(cmd.encode("ascii"), (UNITY_IP, CMD_PORT))
            resp, _ = self.cmd_sock.recvfrom(1024)
            return resp.decode("ascii").strip()
        except socket.timeout:
            return "timeout"
        except OSError:
            # Windows UDP: 상대 포트가 닫히면 ConnectionResetError(10054)가 올라온다
            return "disconnected"

    def pump(self):
        """상태 소켓에 쌓인 패킷을 모두 비우고 최신 값만 유지"""
        while True:
            try:
                data, _ = self.state_sock.recvfrom(8192)
            except socket.timeout:
                return
            except OSError:
                return
            try:
                msg = json.loads(data.decode("ascii"))
            except (ValueError, UnicodeDecodeError):
                continue
            if msg.get("type") == "objects":
                self.objects = msg
            else:
                self.state = msg

    def wait_state(self, timeout: float = 2.0):
        """새 상태 패킷이 올 때까지 대기"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.pump()
            if self.state is not None:
                return self.state
            time.sleep(0.02)
        return self.state

    def request_objects(self, timeout: float = 3.0):
        """objects 명령을 보내고 응답을 기다린다"""
        self.objects = None
        deadline = time.time() + timeout
        self.send("objects")
        while time.time() < deadline:
            self.pump()
            if self.objects is not None:
                return self.objects
            time.sleep(0.05)
        return None

    def close(self):
        try:
            self.send("rc 0 0 0 0")
        except OSError:
            pass
        self.cmd_sock.close()
        self.state_sock.close()


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ── 유령(동적 장애물) 회피 파라미터 ────────────────────────────────
GHOST_AVOID_RADIUS = 5.0    # m — 이 거리 안의 유령에 대해 반발력 발생
GHOST_AVOID_GAIN = 55.0     # 반발 rc 세기 (목표 인력과 합쳐지는 퍼텐셜 필드)
GHOST_AVOID_LIFT = 12.0     # 유령 근접 시 살짝 상승시켜 위로 넘어감


def ghost_avoidance(st, yaw):
    """상태 스트림의 유령 위치로부터 반발 rc(드론 로컬 프레임)를 계산.

    각 유령을 반발원으로 보는 퍼텐셜 필드. goto()의 목표 추종(인력) 성분에
    더해져, 드론이 목표로 가면서도 움직이는 유령을 우회하도록 만든다.
    """
    ghosts = st.get("ghosts") or []
    a_lr = a_fb = a_ud = 0.0
    for g in ghosts:
        dx = st["x"] - g["x"]      # 월드상 '유령 → 드론' 벡터
        dz = st["z"] - g["z"]
        dh = math.hypot(dx, dz)
        if dh >= GHOST_AVOID_RADIUS or dh < 1e-3:
            continue
        strength = (GHOST_AVOID_RADIUS - dh) / GHOST_AVOID_RADIUS   # 0..1 (가까울수록 큼)
        ux, uz = dx / dh, dz / dh                                   # away 단위벡터(월드)
        # 월드 → 드론 로컬 (goto의 목표 변환과 동일한 규약)
        llx = ux * math.cos(yaw) - uz * math.sin(yaw)              # 로컬 우측 성분
        llz = ux * math.sin(yaw) + uz * math.cos(yaw)              # 로컬 전방 성분
        a_lr += llx * GHOST_AVOID_GAIN * strength
        a_fb += llz * GHOST_AVOID_GAIN * strength
        a_ud += GHOST_AVOID_LIFT * strength
    return a_lr, a_fb, a_ud


def goto(sim: SimClient, tx: float, ty: float, tz: float,
         tol_xz: float = 0.35, tol_y: float = 0.3, timeout: float = 30.0,
         label: str = "", avoid: bool = True) -> bool:
    """
    P 제어로 월드 좌표 (tx, ty, tz)까지 비행.
    rc 명령은 드론 로컬 프레임이므로 yaw로 월드 오차를 회전 변환한다.
    avoid=True이면 이동 중 유령(동적 장애물)을 퍼텐셜 필드로 우회한다.
    (정밀 작업인 화물 접근/투하 하강에서는 avoid=False로 꺼서 정렬을 지킨다.)
    """
    KP_XZ = 28.0   # rc단위 / m
    KP_Y = 35.0
    MAX_RC = 35    # ≈ 5 m/s (moveSpeed 15 기준)

    deadline = time.time() + timeout
    while time.time() < deadline:
        st = sim.wait_state()
        if st is None:
            print("  [!] 상태 수신 실패 — 유니티가 Play 중인지 확인")
            return False

        dx = tx - st["x"]
        dy = ty - st["y"]
        dz = tz - st["z"]
        dist_xz = math.hypot(dx, dz)

        if dist_xz < tol_xz and abs(dy) < tol_y:
            sim.send("rc 0 0 0 0")
            return True

        # 월드 → 드론 로컬 (Unity, Y축 yaw)
        yaw = math.radians(st["yaw"])
        lx = dx * math.cos(yaw) - dz * math.sin(yaw)   # 로컬 우측 성분
        lz = dx * math.sin(yaw) + dz * math.cos(yaw)   # 로컬 전방 성분

        # 목표 인력(P 제어) 성분
        cmd_lr = lx * KP_XZ
        cmd_fb = lz * KP_XZ
        cmd_ud = dy * KP_Y

        # 유령(동적 장애물) 반발 성분 — 이동 중에만
        if avoid:
            a_lr, a_fb, a_ud = ghost_avoidance(st, yaw)
            cmd_lr += a_lr
            cmd_fb += a_fb
            cmd_ud += a_ud

        lr = int(clamp(cmd_lr, -MAX_RC, MAX_RC))
        fb = int(clamp(cmd_fb, -MAX_RC, MAX_RC))
        ud = int(clamp(cmd_ud, -MAX_RC, MAX_RC))

        sim.send(f"rc {lr} {fb} {ud} 0")
        time.sleep(0.05)

    print(f"  [!] 이동 시간 초과: {label} (목표 {tx:.1f},{ty:.1f},{tz:.1f})")
    sim.send("rc 0 0 0 0")
    return False


def ascend_to(sim: SimClient, target_y: float, timeout: float = 10.0) -> float:
    """현재 위치에서 수직 상승 (천장 등으로 막히면 도달 가능한 고도 반환)"""
    deadline = time.time() + timeout
    last_y = None
    stall = 0
    while time.time() < deadline:
        st = sim.wait_state()
        if st is None:
            break
        if st["y"] >= target_y - 0.2:
            break
        if last_y is not None and st["y"] - last_y < 0.02:
            stall += 1
            if stall > 25:      # 더 이상 상승 불가 (천장)
                break
        else:
            stall = 0
        last_y = st["y"]
        sim.send("rc 0 0 50 0")
        time.sleep(0.05)
    sim.send("rc 0 0 0 0")
    time.sleep(0.2)
    st = sim.wait_state()
    return st["y"] if st is not None else target_y


def grab_with_retry(sim: SimClient, prop_name: str, px: float, py: float, pz: float,
                    attempts: int = 4) -> bool:
    """화물 직상공에서 수직 하강하며 grab 시도"""
    hover = py + 1.2
    for attempt in range(attempts):
        goto(sim, px, hover, pz, tol_xz=0.25, tol_y=0.2,
             label=f"{prop_name} 접근", avoid=False)
        sim.send("rc 0 0 0 0")
        time.sleep(0.3)
        sim.send("grab")
        time.sleep(0.4)
        st = sim.wait_state()
        if st is not None and st.get("carrying") == prop_name:
            return True
        # 다른 화물을 잡았을 수도 있다 — 그것도 성공으로 처리
        if st is not None and st.get("carrying"):
            print(f"  [i] 목표와 다른 화물을 잡음: {st['carrying']}")
            return True
        hover = max(py + 0.6, hover - 0.25)
        print(f"  [i] grab 재시도 {attempt + 1}/{attempts} (고도 {hover:.2f})")
    return False


def main():
    print("=" * 60)
    print("  자율 화물 배달 미션 (Unity Tello 시뮬레이터)")
    print("=" * 60)

    sim = SimClient()

    # ── 연결 확인 ──────────────────────────────────────────────
    resp = sim.send("command")
    if resp != "ok":
        print(f"[ERROR] 유니티 응답 없음 ('{resp}') — Unity Play 상태인지 확인하세요.")
        sim.close()
        return
    print("[OK] 유니티 연결 성공")

    # ── 오브젝트 목록 수신 ─────────────────────────────────────
    objects = sim.request_objects()
    if objects is None or not objects.get("zones"):
        print("[ERROR] 오브젝트 정보를 받지 못했습니다. SimPropManager가 로드됐는지 확인하세요.")
        sim.close()
        return

    zone = objects["zones"][0]
    targets = [p for p in objects["props"] if not p["delivered"]]
    print(f"[OK] 화물 {len(targets)}개, 배달구역 '{zone['name']}' "
          f"({zone['x']:.1f}, {zone['z']:.1f})")

    if not targets:
        print("배달할 화물이 없습니다.")
        sim.close()
        return

    # ── 동적 장애물(유령) 배치 ─────────────────────────────────
    # 자율 배달 중 움직이는 유령을 퍼텐셜 필드로 우회하는 것을 보여주기 위해
    # 씬에 유령 2기를 임포트한다. (이미 씬에 있으면 그만큼 더 늘어난다)
    print("[+] 유령 2기 임포트 — 자율 회피 대상")
    sim.send("ghost 2")
    time.sleep(0.5)

    # ── 이륙 후 순항고도 확보 ──────────────────────────────────
    # 환경(스캔 메시)이 5배 스케일이라 가구가 3~5m 높이다. 가구 위·천장 아래의
    # 고고도로 순항하고, 목표 직상공에서만 수직으로 내려간다.
    print("\n[1] 이륙 및 순항고도 상승")
    sim.send("takeoff")
    time.sleep(1.0)
    cruise_y = ascend_to(sim, zone["y"] + 9.0)
    cruise_y = max(cruise_y - 0.5, zone["y"] + 2.5)   # 천장에서 0.5m 여유
    print(f"  순항고도: {cruise_y:.1f}m")

    success = 0
    # ── 배달 루프 ──────────────────────────────────────────────
    for i, prop in enumerate(targets):
        name = prop["name"]
        print(f"\n[2.{i + 1}] '{name}' 운반 시작")

        # 최신 위치 갱신 (물리로 굴러갔을 수 있음)
        objs = sim.request_objects()
        if objs is not None:
            cur = next((p for p in objs["props"] if p["name"] == name), None)
            if cur is None:
                continue
            if cur["delivered"]:
                print("  [i] 이미 배달됨 — 건너뜀")
                success += 1
                continue
            prop = cur

        # 순항고도로 수직 상승 → 화물 직상공까지 고고도 이동 → 수직 하강
        st = sim.wait_state()
        goto(sim, st["x"], cruise_y, st["z"], label="순항고도 상승")
        goto(sim, prop["x"], cruise_y, prop["z"], tol_xz=0.35, timeout=40,
             label=f"{name} 상공 이동")

        # 잡기 (직상공에서 수직 하강)
        if not grab_with_retry(sim, name, prop["x"], prop["y"], prop["z"]):
            print(f"  [!] '{name}' 잡기 실패 — 다음 화물로")
            st = sim.wait_state()
            goto(sim, st["x"], cruise_y, st["z"], label="순항고도 복귀")
            continue
        print(f"  [OK] '{name}' 픽업 완료")

        # 수직 상승 후 고고도로 배달구역 직상공까지 이동
        st = sim.wait_state()
        goto(sim, st["x"], cruise_y, st["z"], label="운반고도 상승")
        goto(sim, zone["x"], cruise_y, zone["z"], tol_xz=0.35, timeout=40,
             label="배달구역 상공 이동")

        # 패드 위로 수직 하강. 화물은 이제 드론 배면에 딱 붙어(리지드 클램프) 운반되고,
        # drop 시 바로 아래 표면에 스냅되어 놓인다. 드론을 패드 가까이 내린 뒤 투하한다.
        drop_y = zone["y"] + 1.2
        on_target = goto(sim, zone["x"], drop_y, zone["z"], tol_xz=0.25,
                         label="패드 위 하강", avoid=False)
        if not on_target:
            # 한 번 더: 다시 올라가서 재접근
            print("  [i] 패드 재접근 시도")
            st = sim.wait_state()
            goto(sim, st["x"], cruise_y, st["z"], label="재상승")
            goto(sim, zone["x"], cruise_y, zone["z"], tol_xz=0.3, label="상공 재정렬")
            on_target = goto(sim, zone["x"], drop_y, zone["z"], tol_xz=0.25,
                             label="패드 위 재하강", avoid=False)

        # 놓기 (투하 직전의 배달 수를 기준으로 성공 판정)
        sim.send("rc 0 0 0 0")
        time.sleep(0.3)
        st = sim.wait_state()
        before = st.get("delivered", 0) if st else 0
        if st is not None:
            off = math.hypot(st["x"] - zone["x"], st["z"] - zone["z"])
            print(f"  [→] 화물 투하 (패드 중심에서 {off:.2f}m), 착지 대기...")
        else:
            print("  [→] 화물 투하, 착지 대기...")
        sim.send("drop")

        # 배달 확인 (최대 6초)
        ok = False
        deadline = time.time() + 6.0
        while time.time() < deadline:
            st = sim.wait_state()
            if st is not None and st.get("delivered", 0) > before:
                ok = True
                break
            time.sleep(0.2)
        if ok:
            success += 1
            print(f"  [OK] 배달 완료! ({success}/{len(targets)})")
        else:
            print("  [!] 배달 확인 실패")
            objs = sim.request_objects()
            if objs is not None:
                cur = next((p for p in objs["props"] if p["name"] == name), None)
                if cur is not None:
                    off = math.hypot(cur["x"] - zone["x"], cur["z"] - zone["z"])
                    print(f"      → '{name}' 실제 위치: ({cur['x']:.2f}, {cur['y']:.2f}, "
                          f"{cur['z']:.2f}) — 패드 중심에서 {off:.2f}m")

    # ── 착륙 및 리포트 ─────────────────────────────────────────
    print("\n[3] 귀환 및 착륙")
    st = sim.wait_state()
    if st is not None:
        goto(sim, st["x"], cruise_y, st["z"], label="상승")
    sim.send("rc 0 0 0 0")
    time.sleep(0.2)
    sim.send("land")

    st = sim.wait_state()
    print("\n" + "=" * 60)
    print("  미션 리포트")
    print("=" * 60)
    print(f"  배달 성공 : {success} / {len(targets)}")
    if st is not None:
        print(f"  충돌 횟수 : {st.get('collision_count', '?')}")
        print(f"  최종 위치 : ({st['x']:.2f}, {st['y']:.2f}, {st['z']:.2f})")
    print("=" * 60)

    sim.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[중단] 정지 명령 전송 후 종료")
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(b"rc 0 0 0 0", (UNITY_IP, CMD_PORT))
        s.sendto(b"land", (UNITY_IP, CMD_PORT))
        s.close()
