import cv2
import numpy as np
import threading
import time
from djitellopy import Tello

# ══════════════════════════════════════════════════════
# 설정값
# ══════════════════════════════════════════════════════
CANVAS_W       = 700
CANVAS_H       = 700
PIXELS_PER_CM  = 5.0      # 1cm = 5픽셀 (그림 크기 결정)
FLIGHT_SPEED   = 40       # RC 속도 (0~100)
RC_INTERVAL    = 0.05     # RC 명령 전송 주기 (초)
SAMPLE_DIST    = 100      # 웨이포인트 샘플링 간격 (픽셀)
HOVER_SECS     = 2.0      # 이륙 후 안정화 시간 (초)


# ══════════════════════════════════════════════════════
# 경로 샘플러
# ══════════════════════════════════════════════════════
def sample_path(points: list, dist: int) -> list:
    if len(points) < 2:
        return list(points)

    result = [points[0]]
    accum  = 0.0

    for i in range(1, len(points)):
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        accum += np.hypot(dx, dy)
        if accum >= dist:   
            result.append(points[i])
            accum = 0.0

    if result[-1] != points[-1]:
        result.append(points[-1])

    return result


# ══════════════════════════════════════════════════════
# RC 헬퍼
# ══════════════════════════════════════════════════════
def hover(drone, secs):
    end_t = time.time() + secs
    while time.time() < end_t:
        drone.send_rc_control(0, 0, 0, 0)
        time.sleep(RC_INTERVAL)


def fly_segment_xz(drone, dx_px, dy_px):
    """
    XZ 평면 이동.
      dx_px 양수 = 화면 오른쪽 → 드론 우측 (lr+)
      dy_px 양수 = 화면 아래   → 드론 하강 (ud-)
      dy_px 음수 = 화면 위     → 드론 상승 (ud+)
    """
    mag = np.hypot(dx_px, dy_px)
    if mag < 1:
        return

    dist_cm = mag / PIXELS_PER_CM
    ux      =  dx_px / mag   # 좌우 성분
    uz      = -dy_px / mag   # 상하 성분 (화면 아래 = 드론 하강 → 반전)

    rc_lr = int(np.clip(ux * FLIGHT_SPEED, -100, 100))
    rc_ud = int(np.clip(uz * FLIGHT_SPEED, -100, 100))

    duration = dist_cm / max(FLIGHT_SPEED * 0.8, 1.0)

    end_t = time.time() + duration
    while time.time() < end_t:
        drone.send_rc_control(rc_lr, 0, rc_ud, 0)   # fb=0, yaw=0 고정
        time.sleep(RC_INTERVAL)


# ══════════════════════════════════════════════════════
# 비행 스레드
# ══════════════════════════════════════════════════════
def fly_path(drone: Tello, waypoints: list, status: dict):
    status["flying"]  = True
    status["current"] = 0
    status["done"]    = False

    print("[드론] 이륙")
    drone.takeoff()
    hover(drone, HOVER_SECS)

    total = len(waypoints) - 1
    for i in range(total):
        if not status["flying"]:
            break

        status["current"] = i + 1
        p0 = waypoints[i]
        p1 = waypoints[i + 1]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]

        dist_cm = np.hypot(dx, dy) / PIXELS_PER_CM
        lr_cm   =  dx / PIXELS_PER_CM
        ud_cm   = -dy / PIXELS_PER_CM   # 화면 위 = 상승 = 양수

        print(f"  [{i+1}/{total}]  "
              f"좌우={lr_cm:+.1f}cm  상하={ud_cm:+.1f}cm  "
              f"(거리={dist_cm:.1f}cm)")

        fly_segment_xz(drone, dx, dy)
        hover(drone, 0.2)

    hover(drone, 0.5)
    drone.send_rc_control(0, 0, 0, 0)
    time.sleep(0.5)
    drone.land()

    status["flying"] = False
    status["done"]   = True
    print("[드론] 착륙 완료!")


# ══════════════════════════════════════════════════════
# UI 렌더러
# ══════════════════════════════════════════════════════
def render(canvas, waypoints, status):
    ui = canvas.copy()

    # 배경 격자
    for x in range(0, CANVAS_W, 50):
        cv2.line(ui, (x, 0), (x, CANVAS_H), (28, 28, 28), 1)
    for y in range(0, CANVAS_H, 50):
        cv2.line(ui, (0, y), (CANVAS_W, y), (28, 28, 28), 1)

    # 축 라벨 (화면 가장자리)
    cv2.putText(ui, "LEFT",  (8, CANVAS_H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    cv2.putText(ui, "RIGHT", (CANVAS_W-50, CANVAS_H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    cv2.putText(ui, "UP",    (CANVAS_W//2-10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
    cv2.putText(ui, "DOWN",  (CANVAS_W//2-15, CANVAS_H-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # 중심 십자선
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    cv2.line(ui, (cx-15, cy), (cx+15, cy), (50, 50, 50), 1)
    cv2.line(ui, (cx, cy-15), (cx, cy+15), (50, 50, 50), 1)

    # 경로 선 & 웨이포인트
    if len(waypoints) > 1:
        for i in range(1, len(waypoints)):
            cv2.line(ui, waypoints[i-1], waypoints[i], (0, 170, 70), 2)

        for i, wp in enumerate(waypoints):
            if status["flying"] and i == status["current"]:
                cv2.circle(ui, wp, 9,  (0, 80, 255), -1)
                cv2.circle(ui, wp, 13, (0, 150, 255), 2)
            else:
                color = (0, 255, 100) if i == 0 else (80, 200, 255)
                cv2.circle(ui, wp, 4, color, -1)

        cv2.circle(ui, waypoints[0],  10, (0, 255, 0),   2)
        cv2.circle(ui, waypoints[-1], 10, (0, 100, 255), 2)

        # 방향 화살표
        for i in range(1, len(waypoints)):
            if i % 5 == 0:
                p0  = waypoints[i-1]
                p1  = waypoints[i]
                mx  = (p0[0]+p1[0])//2
                my  = (p0[1]+p1[1])//2
                ang = np.arctan2(p1[1]-p0[1], p1[0]-p0[0])
                ax  = int(mx + 8*np.cos(ang))
                ay  = int(my + 8*np.sin(ang))
                cv2.arrowedLine(ui, (mx,my), (ax,ay),
                                (0,140,60), 2, tipLength=0.8)

    # ── 안내 패널 ──────────────────────────────────
    panel = np.zeros((CANVAS_H, 210, 3), dtype=np.uint8)

    def T(msg, row, color=(160, 160, 160)):
        cv2.putText(panel, msg, (10, 24 + row * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

    T("DRONE  DRAW  XZ",    0, (0, 210, 90))
    T("[ Front View ]",         1, (60, 160, 80))
    T("=" * 24,              2, (45, 45, 45))
    T("DRAG  : Draw",  3)
    T("SPACE : Flight Start",    4)
    T("R     : reset",       5)
    T("ESC   : exit",         6)
    T("=" * 24,              7, (45, 45, 45))

    wp = len(waypoints)
    if status["flying"]:
        T(">>> Flighting <<<",                      11, (0, 200, 255))
        T(f"{status['current']}/{max(wp-1,1)} wp", 12, (0, 170, 210))
    elif status["done"]:
        T("done!",       11, (0, 255, 100))
        T("R: reset", 12)
    elif wp > 1:
        total_dist = sum(
            np.hypot(waypoints[i][0]-waypoints[i-1][0],
                     waypoints[i][1]-waypoints[i-1][1])
            for i in range(1, wp)
        ) / PIXELS_PER_CM
        T(f"WP: {wp}개",                  11, (190, 190, 80))
        T(f"distance: {total_dist:.0f}cm",    12, (170, 170, 70))
        T("SPACE = flight",                 13, (0, 210, 90))

    return np.hstack([ui, panel])


# ══════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════
def main():
    Tello.CONTROL_UDP_PORT_CLIENT = 9000
    drone = Tello("127.0.0.1")
    drone.connect()
    print(f"[연결] 배터리: {drone.get_battery()}%\n")
    print("[ XZ 평면 모드 ] 그린 그림이 정면에서 보입니다.\n")

    canvas    = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    raw_pts   = []
    waypoints = []
    drawing   = False
    status    = {"flying": False, "current": 0, "done": False}

    def on_mouse(event, x, y, flags, param):
        nonlocal drawing, raw_pts, waypoints, canvas

        if status["flying"]:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing   = True
            raw_pts   = [(x, y)]
            canvas    = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            waypoints = []
            status["done"] = False

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            raw_pts.append((x, y))
            if len(raw_pts) > 1:
                cv2.line(canvas, raw_pts[-2], raw_pts[-1], (40, 40, 40), 1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing   = False
            waypoints = sample_path(raw_pts, SAMPLE_DIST)
            total_cm  = sum(
                np.hypot(waypoints[i][0]-waypoints[i-1][0],
                         waypoints[i][1]-waypoints[i-1][1])
                for i in range(1, len(waypoints))
            ) / PIXELS_PER_CM
            print(f"[경로] {len(raw_pts)}점 → {len(waypoints)}개 웨이포인트 / 총 {total_cm:.0f}cm")

    cv2.namedWindow("Drone Draw XZ")
    cv2.setMouseCallback("Drone Draw XZ", on_mouse)

    print("마우스로 경로를 그리세요.")
    print("SPACE=비행  R=초기화  ESC=종료\n")

    while True:
        frame = render(canvas, waypoints, status)
        cv2.imshow("Drone Draw XZ", frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:                       # ESC
            if status["flying"]:
                status["flying"] = False
                drone.send_rc_control(0, 0, 0, 0)
                time.sleep(0.3)
                drone.land()
            break

        elif key in (ord("r"), ord("R")):   # R: 초기화
            if not status["flying"]:
                raw_pts   = []
                waypoints = []
                canvas    = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                status["done"] = False
                print("[초기화]")

        elif key == ord(" "):               # SPACE: 비행
            if status["flying"]:
                print("[경고] 이미 비행 중입니다")
            elif len(waypoints) < 2:
                print("[경고] 경로를 먼저 그리세요")
            else:
                threading.Thread(
                    target=fly_path,
                    args=(drone, waypoints, status),
                    daemon=True
                ).start()

    cv2.destroyAllWindows()
    drone.end()
    print("[종료]")


if __name__ == "__main__":
    main()