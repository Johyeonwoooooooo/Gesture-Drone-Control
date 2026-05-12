import socket
import time
import pygame

# ──────────────────────────────────────────────────────────────────────────────
# djitellopy 없이 직접 UDP로 통신하는 방식 사용
# (djitellopy 내부 동작 문제를 완전히 우회)
# ──────────────────────────────────────────────────────────────────────────────

UNITY_IP   = "127.0.0.1"
UNITY_PORT = 9000       # 유니티 수신 포트
LOCAL_PORT = 9001       # 이 스크립트의 수신 포트 (응답 받는 포트)

def create_socket():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", LOCAL_PORT))
    sock.settimeout(0.1)  # 응답 대기 최대 0.1초
    return sock

def send_command(sock, cmd: str):
    """유니티로 명령 전송 후 응답 수신"""
    sock.sendto(cmd.encode("ascii"), (UNITY_IP, UNITY_PORT))
    try:
        response, _ = sock.recvfrom(1024)
        return response.decode("ascii").strip()
    except socket.timeout:
        return "timeout"

def main():
    print("=" * 50)
    print("  Tello 시뮬레이터 컨트롤러")
    print("=" * 50)
    print(f"  유니티 주소: {UNITY_IP}:{UNITY_PORT}")
    print(f"  로컬 포트:   {LOCAL_PORT}")
    print()
    print("[조종 방법]")
    print("  T       : 이륙 (Takeoff)")
    print("  L       : 착륙 (Land)")
    print("  W / S   : 전진 / 후진")
    print("  A / D   : 좌이동 / 우이동")
    print("  R / F   : 상승 / 하강")
    print("  Q / E   : 좌회전 / 우회전")
    print("  ESC     : 종료")
    print("=" * 50)

    # 소켓 생성
    try:
        sock = create_socket()
        print(f"\n[OK] 소켓 생성 완료 (로컬 포트 {LOCAL_PORT})")
    except Exception as e:
        print(f"[ERROR] 소켓 생성 실패: {e}")
        print(f"  → 포트 {LOCAL_PORT}이 이미 사용 중일 수 있습니다.")
        return

    # 유니티에 초기화 신호 전송
    print("\n유니티 시뮬레이터에 연결 시도...")
    resp = send_command(sock, "command")
    if resp == "ok":
        print("[OK] 유니티 연결 성공!")
    else:
        print(f"[WARNING] 응답: '{resp}' — 유니티가 실행 중인지 확인하세요.")
        print("  → Unity에서 Play 버튼을 먼저 누르세요!\n")

    # Pygame 초기화
    pygame.init()
    win = pygame.display.set_mode((400, 300))
    pygame.display.set_caption("Tello Simulator Controller — 이 창 클릭 후 조종")

    font = pygame.font.SysFont("malgun gothic", 18)  # 한글 폰트

    speed   = 50   # RC 값 (-100 ~ 100)
    running = True
    is_flying = False

    clock = pygame.time.Clock()

    print("\n[INFO] Pygame 창을 클릭한 후 키를 누르세요!\n")

    while running:
        # ── 이벤트 처리 ──────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_t and not is_flying:
                    resp = send_command(sock, "takeoff")
                    print(f"[이륙] 전송 → 응답: {resp}")
                    if resp == "ok":
                        is_flying = True

                elif event.key == pygame.K_l and is_flying:
                    resp = send_command(sock, "land")
                    print(f"[착륙] 전송 → 응답: {resp}")
                    if resp == "ok":
                        is_flying = False

        # ── RC 값 계산 ───────────────────────────────────────────────────────
        keys = pygame.key.get_pressed()

        lr, fb, ud, yaw = 0, 0, 0, 0

        if keys[pygame.K_a]:      lr  = -speed   # 좌이동
        elif keys[pygame.K_d]:    lr  =  speed   # 우이동

        if keys[pygame.K_w]:      fb  =  speed   # 전진
        elif keys[pygame.K_s]:    fb  = -speed   # 후진

        if keys[pygame.K_r]:      ud  =  speed   # 상승
        elif keys[pygame.K_f]:    ud  = -speed   # 하강

        if keys[pygame.K_q]:      yaw = -speed   # 좌회전
        elif keys[pygame.K_e]:    yaw =  speed   # 우회전

        # ── RC 명령 전송 (비행 중일 때만) ────────────────────────────────────
        if is_flying:
            cmd = f"rc {lr} {fb} {ud} {yaw}"
            send_command(sock, cmd)

        # ── 화면 렌더링 ──────────────────────────────────────────────────────
        win.fill((30, 30, 30))

        status_color = (0, 220, 255) if is_flying else (255, 200, 0)
        status_text  = "비행 중 ✈" if is_flying else "착륙 상태"

        lines = [
            (f"상태: {status_text}",          status_color),
            (f"LR:{lr:+4d}  FB:{fb:+4d}",    (200, 200, 200)),
            (f"UD:{ud:+4d}  Yaw:{yaw:+4d}",  (200, 200, 200)),
            ("",                               (255,255,255)),
            ("T:이륙  L:착륙  ESC:종료",       (150, 150, 150)),
            ("WASD:이동  RF:상하  QE:회전",    (150, 150, 150)),
        ]

        for i, (text, color) in enumerate(lines):
            surf = font.render(text, True, color)
            win.blit(surf, (20, 20 + i * 28))

        pygame.display.flip()

        # 20Hz
        clock.tick(20)

    # ── 종료 처리 ────────────────────────────────────────────────────────────
    if is_flying:
        print("\n[INFO] 종료 전 착륙 명령 전송...")
        send_command(sock, "land")
        time.sleep(0.3)

    sock.close()
    pygame.quit()
    print("[INFO] 종료 완료.")

if __name__ == "__main__":
    main()