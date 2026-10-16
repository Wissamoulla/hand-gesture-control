import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# Pairs of landmark indices to connect with lines (the hand "skeleton")
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
]

# Set up the hand landmark model
options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the camera")
    exit()

start_time = time.time()

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read a frame")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # MediaPipe expects RGB, OpenCV gives BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # VIDEO mode needs an increasing timestamp in milliseconds
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    for hand in result.hand_landmarks:
        # Landmarks are normalized (0..1), convert them to pixel positions
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

        for i, (x, y) in enumerate(points):
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, str(i), (x + 6, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
