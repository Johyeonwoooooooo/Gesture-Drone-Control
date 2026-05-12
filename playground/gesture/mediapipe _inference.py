import cv2
import mediapipe as mp

# mediapipe hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR -> RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # inference
    results = hands.process(image)

    # ======================
    # mediapipe output 확인
    # ======================
    print("------ results ------")
    print(results)

    if results.multi_hand_landmarks:
        print("손 개수:", len(results.multi_hand_landmarks))

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

            print(f"\n손 {hand_idx}")

            for i, lm in enumerate(hand_landmarks.landmark):
                print(f"landmark {i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

            # 랜드마크 그리기
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    if results.multi_handedness:
        for handedness in results.multi_handedness:
            print("Left / Right:", handedness.classification[0].label)
            print("Confidence:", handedness.classification[0].score)

    cv2.imshow("hand", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()