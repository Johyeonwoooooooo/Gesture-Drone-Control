# -*- coding: utf-8 -*-
"""
gesture_cam.py — 웹캠 제스처 인식 + 미리보기 스트림
────────────────────────────────────────────────────────────────────
`playground/gesture/Gesture_Drone_control_test.py` 의 인식 규칙
(`fingers_up` / `get_gesture`)을 그대로 재사용한다. 다른 점은 두 가지:

  · 드론에 직접 rc 를 쏘지 않고 **현재 제스처와 rc 벡터를 상태로만** 들고 있다.
    누가 가져다 쓸지는 호출부가 정한다(유령 근접 시 개입 등).
  · 랜드마크를 그린 프레임을 JPEG 으로 계속 갈아 끼워, 웹이
    `/api/camera` (MJPEG) 로 우하단에 띄울 수 있게 한다.

웹캠은 보통 한 프로세스만 열 수 있다. 그래서 **브라우저가 직접 카메라를
잡지 않고** 여기서 잡은 화면을 넘겨준다. 랜드마크·인식 결과가 그려진
화면이라 데모에도 이쪽이 낫다.

    import gesture_cam
    gesture_cam.start(0)              # 카메라 인덱스
    gesture_cam.state()               # {'gesture': '상승 ↑', 'rc': [0,0,40,0], ...}
    gesture_cam.latest_jpeg()         # bytes | None

카메라나 mediapipe 가 없어도 죽지 않는다 — active=False 로 조용히 꺼진다.
"""
from __future__ import annotations

import os
import sys
import time
import threading

_ROOT = os.path.dirname(os.path.abspath(__file__))
_GESTURE = os.path.join(_ROOT, 'playground', 'gesture')
if _GESTURE not in sys.path:
    sys.path.insert(0, _GESTURE)

MOVE_SPEED = 40          # 원본 스크립트와 같은 rc 크기
JPEG_QUALITY = 70
IDLE_GESTURE = '정지 ✋'

# 제스처 -> (좌우, 전후, 상하, 요) — send_rc_control 규약과 동일
RC_MAP = {
    '상승 ↑':   (0, 0, MOVE_SPEED, 0),
    '하강 ↓':   (0, 0, -MOVE_SPEED, 0),
    '왼쪽 ←':   (-MOVE_SPEED, 0, 0, 0),
    '오른쪽 →': (MOVE_SPEED, 0, 0, 0),
    '앞으로 ▲': (0, MOVE_SPEED, 0, 0),
    '뒤로 ▽':   (0, -MOVE_SPEED, 0, 0),
    '회전 ↻':   (0, 0, 0, MOVE_SPEED),
    IDLE_GESTURE: (0, 0, 0, 0),
}

_lock = threading.Lock()
_thread = None
_stop = threading.Event()
_frame = None            # 최신 JPEG bytes
_gesture = None
_fps = 0.0
_error = None
_active = False


def state():
    """현재 인식 상태. 카메라가 없으면 active=False."""
    with _lock:
        g = _gesture
        return {
            'active': _active,
            'gesture': g,
            'rc': list(RC_MAP.get(g, (0, 0, 0, 0))),
            'fps': round(_fps, 1),
            'error': _error,
            'has_frame': _frame is not None,
        }


def latest_jpeg():
    with _lock:
        return _frame


def _loop(cam_index: int, width: int, height: int):
    global _frame, _gesture, _fps, _error, _active
    try:
        import cv2
        import mediapipe as mp
        from gesture_rules import fingers_up, get_gesture   # 기존 인식 규칙 그대로 재사용
    except Exception as e:                       # mediapipe/opencv 없음
        with _lock:
            _error = f'제스처 모듈 로드 실패: {e}'
            _active = False
        return

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        with _lock:
            _error = f'카메라 {cam_index} 를 열 수 없습니다 (다른 앱이 쓰는 중일 수 있음)'
            _active = False
        return

    mp_hands, mp_draw = mp.solutions.hands, mp.solutions.drawing_utils
    with _lock:
        _active, _error = True, None
    print(f'[gesture_cam] 카메라 {cam_index} 시작 ({width}x{height})')

    last, ema = time.time(), 0.0
    try:
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7,
                            min_tracking_confidence=0.5) as hands:
            while not _stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)       # 거울 모드 (원본과 동일)
                res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                g = None
                if res.multi_hand_landmarks:
                    hl = res.multi_hand_landmarks[0]
                    handed = res.multi_handedness[0].classification[0].label
                    g = get_gesture(fingers_up(hl, handed))
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                # 한글은 cv2 로 못 그려서 화살표/기호만 남긴다(폰트 의존 제거).
                tag = (g or '-').split(' ')[-1] if g else '-'
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (12, 6, 10), -1)
                cv2.putText(frame, f'GESTURE {tag}', (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (120, 255, 190) if g else (110, 110, 120), 2)

                ok2, buf = cv2.imencode('.jpg', frame,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                now = time.time()
                ema = 0.9 * ema + 0.1 / max(1e-3, now - last)
                last = now
                with _lock:
                    if ok2:
                        _frame = buf.tobytes()
                    _gesture = g
                    _fps = ema
    finally:
        cap.release()
        with _lock:
            _active = False
        print('[gesture_cam] 카메라 종료')


def start(cam_index: int = 0, width: int = 480, height: int = 360) -> bool:
    """카메라 스레드를 띄운다. 이미 돌고 있으면 무시."""
    global _thread
    if _thread and _thread.is_alive():
        return True
    _stop.clear()
    _thread = threading.Thread(target=_loop, args=(cam_index, width, height), daemon=True)
    _thread.start()
    for _ in range(60):                      # 최대 3초까지 첫 프레임 대기
        time.sleep(0.05)
        st = state()
        if st['active'] or st['error']:
            break
    st = state()
    if st['error']:
        print(f"[gesture_cam] {st['error']}")
    return st['active']


def stop():
    _stop.set()
    if _thread:
        _thread.join(timeout=2.0)


if __name__ == '__main__':                   # 간이 확인: 5초간 인식 결과 출력
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if start(idx):
        for _ in range(10):
            time.sleep(0.5)
            s = state()
            print(f"  gesture={s['gesture']!r} rc={s['rc']} fps={s['fps']} "
                  f"frame={'있음' if s['has_frame'] else '없음'}")
        stop()
    else:
        print('카메라를 쓸 수 없습니다:', state()['error'])
