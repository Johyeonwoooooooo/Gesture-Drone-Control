# -*- coding: utf-8 -*-
"""
gesture_rules.py — 손 랜드마크 -> 제스처 판정 (순수 함수)

`Gesture_Drone_control_test.py` 에 있던 두 함수를 그대로 옮긴 것.
그 스크립트는 임포트만 해도 모듈 최상단에서 Tello 에 연결을 시도하기
때문에, 인식 규칙만 필요한 쪽(웹 서버의 gesture_cam 등)이 가져다 쓸 수
있도록 분리했다. 판정 기준은 손바닥이 카메라를 향하고 좌우 반전(flip)된
프레임을 전제로 한다.
"""


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
