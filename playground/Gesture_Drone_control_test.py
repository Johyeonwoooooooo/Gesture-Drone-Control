import cv2
import mediapipe as mp
from djitellopy import Tello
import time
import threading

# ========================
# Tello 시뮬레이터 연결
# ========================
Tello.CONTROL_UDP_PORT_CLIENT = 9000   # sketch_to_flight.py 와 동일한 시뮬레이터 포트
print("start")
tello = Tello("127.0.0.1")
print("created")

tello.connect()
print("connected")
tello.streamoff()
tello.streamon()
print(f"[시작] 배터리 {tello.get_battery()}% | 제스처 조종을 시작합니다.")
tello.takeoff()

# 드론 카메라 프레임 리더
drone_frame_reader = tello.get_frame_read()

# ========================
# MediaPipe 설정
# ========================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ========================
# 드론 제어 설정
# ========================
MOVE_SPEED = 30  # 이동 속도 (0~100)

current_gesture = None
gesture_lock = threading.Lock()
stop_event = threading.Event()


def fingers_up(hand_landmarks, handedness):
    """
    손바닥이 카메라를 향한 기준으로 각 손가락 펴짐 판단
    반환: [thumb, index, middle, ring, pinky]
    """
    fingers = []
    lm = hand_landmarks.landmark

    # 엄지: 손바닥 기준, flip 후 Right → 엄지가 왼쪽(x 작음)
    if handedness == 'Right':
        fingers.append(lm[4].x < lm[3].x)
    else:
        fingers.append(lm[4].x > lm[3].x)

    # 검지~새끼: tip y < pip y 이면 펴진 것
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        fingers.append(lm[tip].y < lm[pip].y)

    return fingers  # [thumb, index, middle, ring, pinky]


def get_gesture(fingers):
    """
    손가락 조합으로 제스처 반환
    """
    thumb, index, middle, ring, pinky = fingers

    if index and not thumb and not middle and not ring and not pinky:
        return "상승 ↑"
    elif index and middle and not thumb and not ring and not pinky:
        return "하강 ↓"
    elif thumb and not index and not middle and not ring and not pinky:
        return "왼쪽 ←"
    elif pinky and not thumb and not index and not middle and not ring:
        return "오른쪽 →"
    elif thumb and pinky and not index and not middle and not ring:
        return "회전 ↻"
    elif thumb and index and not middle and not ring and not pinky:
        return "앞으로 ▲"
    elif thumb and index and middle and ring and pinky:
        return "뒤로 ▽"
    elif not any(fingers):
        return "정지 ✋"
    else:
        return None


def drone_control_loop():
    """
    20Hz로 현재 제스처를 드론 RC 명령으로 전송
    정지 or 손 없음 → rc 0 0 0 0 (Tello 내부 호버링에 맡김)
    """
    while not stop_event.is_set():
        with gesture_lock:
            gesture = current_gesture

        if gesture and gesture != "정지 ✋":
            if gesture == "상승 ↑":
                tello.send_rc_control(0, 0, MOVE_SPEED, 0)
            elif gesture == "하강 ↓":
                tello.send_rc_control(0, 0, -MOVE_SPEED, 0)
            elif gesture == "왼쪽 ←":
                tello.send_rc_control(-MOVE_SPEED, 0, 0, 0)
            elif gesture == "오른쪽 →":
                tello.send_rc_control(MOVE_SPEED, 0, 0, 0)
            elif gesture == "앞으로 ▲":
                tello.send_rc_control(0, MOVE_SPEED, 0, 0)
            elif gesture == "뒤로 ▽":
                tello.send_rc_control(0, -MOVE_SPEED, 0, 0)
            elif gesture == "회전 ↻":
                tello.send_rc_control(0, 0, 0, MOVE_SPEED)
        else:
            # 정지 or 손 인식 없음 → Tello 내부 센서가 알아서 호버링
            tello.send_rc_control(0, 0, 0, 0)

        time.sleep(0.05)  # 20Hz


def run_camera():
    global current_gesture
    prev_gesture = None

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            gesture_text = "손을 인식 중..."
            gesture_color = (200, 200, 200)

            if result.multi_hand_landmarks and result.multi_handedness:
                for handLms, handInfo in zip(result.multi_hand_landmarks,
                                             result.multi_handedness):
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                    handedness = handInfo.classification[0].label
                    fingers = fingers_up(handLms, handedness)
                    gesture = get_gesture(fingers)

                    # 손가락 상태 디버깅
                    finger_names = ["엄지", "검지", "중지", "약지", "새끼"]
                    debug_text = " | ".join(
                        f"{n}{'O' if v else 'X'}" for n, v in zip(finger_names, fingers)
                    )
                    cv2.putText(frame, debug_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

                    if gesture:
                        gesture_text = gesture
                        gesture_color = (0, 255, 100) if gesture != "정지 ✋" else (0, 200, 255)
                        if gesture != prev_gesture:
                            print(f"[제스처] {gesture}")
                        with gesture_lock:
                            current_gesture = gesture
                    else:
                        gesture_text = "제스처 없음"
                        gesture_color = (100, 100, 255)
                        with gesture_lock:
                            current_gesture = None

                    prev_gesture = gesture

            else:
                with gesture_lock:
                    current_gesture = None
                prev_gesture = None

            # 제스처 결과 표시
            cv2.putText(frame, gesture_text, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, gesture_color, 3)

            # 조작 안내
            guide = [
                "1. index            -> up",
                "2. index+middle     -> down",
                "3. thumb            -> left",
                "4. pinky            -> right",
                "5. thumb+pinky      -> rotate CW",
                "6. thumb+index      -> forward",
                "7. all fingers      -> backward",
                "8. fist             -> stop/hover",
                "ESC: land & quit"
            ]
            for i, g in enumerate(guide):
                cv2.putText(frame, g, (10, frame.shape[0] - 210 + i * 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

            # 제스처 창
            cv2.imshow("Hand Gesture Control", frame)

            # 드론 카메라 창
            drone_frame = drone_frame_reader.frame
            if drone_frame is not None:
                drone_frame = cv2.resize(drone_frame, (960, 720))
                cv2.imshow("Drone Camera", drone_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


# ========================
# 메인 실행
# ========================
ctrl_thread = threading.Thread(target=drone_control_loop, daemon=True)
ctrl_thread.start()

run_camera()

tello.send_rc_control(0, 0, 0, 0)
tello.land()
tello.streamoff()