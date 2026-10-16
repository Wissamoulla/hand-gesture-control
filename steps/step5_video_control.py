import math
import time

import cv2
import mediapipe as mp
import pyautogui
from mediapipe.tasks.python import BaseOptions, vision

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
]

# (tip, middle joint) for the four long fingers
FINGER_JOINTS = [(8, 6), (12, 10), (16, 14), (20, 18)]

GESTURE_LABELS = {
    "OPEN_PALM": "Open Palm",
    "FIST": "Fist",
    "THUMBS_UP": "Thumbs Up",
    "PEACE": "Peace (V)",
    "POINT": "Point",
}

# Which key each gesture presses. Media keys work without focusing the player
# (YouTube in the browser, VLC, Spotify, ...)
GESTURE_ACTIONS = {
    "OPEN_PALM": ("playpause", "Play / Pause"),
    "THUMBS_UP": ("volumemute", "Mute / Unmute"),
}

STABLE_FRAMES = 8      # how many frames in a row a gesture must hold
COOLDOWN_SECONDS = 1.0  # minimum time between two commands


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def get_open_fingers(points):
    """Return a list of 5 booleans: True if that finger is open."""
    wrist = points[0]
    fingers = []

    # Thumb: open if its tip is farther from the pinky base than its middle joint
    pinky_base = points[17]
    fingers.append(distance(points[4], pinky_base) > distance(points[3], pinky_base))

    # Other fingers: open if the tip is farther from the wrist than the middle joint
    for tip, joint in FINGER_JOINTS:
        fingers.append(distance(points[tip], wrist) > distance(points[joint], wrist))

    return fingers


def recognize_gesture(fingers):
    """Turn the 5 open/closed fingers into a gesture name (or None)."""
    thumb = fingers[0]
    others = fingers[1:]  # index, middle, ring, pinky

    if all(fingers):
        return "OPEN_PALM"
    if not any(fingers):
        return "FIST"
    if thumb and not any(others):
        return "THUMBS_UP"
    # The thumb is ignored here because people hold it either way
    if others == [True, True, False, False]:
        return "PEACE"
    if others == [True, False, False, False]:
        return "POINT"
    return None


class GestureTrigger:
    """Fires a gesture once it has been held steady, then waits for a new one."""

    def __init__(self, stable_frames, cooldown_seconds):
        self.stable_frames = stable_frames
        self.cooldown_seconds = cooldown_seconds
        self.current = None       # gesture seen in the last frames
        self.count = 0            # how many frames in a row we've seen it
        self.fired = False        # already triggered for this hold?
        self.last_fire_time = 0.0

    def update(self, gesture):
        """Call once per frame. Returns the gesture when it should fire, else None."""
        if gesture == self.current:
            self.count += 1
        else:
            self.current = gesture
            self.count = 1
            self.fired = False

        is_stable = self.count >= self.stable_frames
        cooled_down = time.time() - self.last_fire_time >= self.cooldown_seconds

        if gesture is not None and is_stable and not self.fired and cooled_down:
            self.fired = True
            self.last_fire_time = time.time()
            return gesture
        return None

    def progress(self):
        """0.0 .. 1.0, how close the current gesture is to being stable."""
        if self.fired or self.current is None:
            return 0.0
        return min(self.count / self.stable_frames, 1.0)


options = vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
detector = vision.HandLandmarker.create_from_options(options)
trigger = GestureTrigger(STABLE_FRAMES, COOLDOWN_SECONDS)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open the camera")
    exit()

start_time = time.time()
last_action = ""
last_action_time = 0.0

while True:
    success, frame = cap.read()
    if not success:
        print("Could not read a frame")
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int((time.time() - start_time) * 1000)
    result = detector.detect_for_video(mp_image, timestamp_ms)

    gesture = None

    for hand in result.hand_landmarks:
        points = [(int(lm.x * width), int(lm.y * height)) for lm in hand]

        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

        fingers = get_open_fingers(points)
        for tip, is_open in zip([4, 8, 12, 16, 20], fingers):
            color = (0, 255, 0) if is_open else (0, 0, 255)
            cv2.circle(frame, points[tip], 10, color, -1)

        gesture = recognize_gesture(fingers)

    fired = trigger.update(gesture)
    if fired in GESTURE_ACTIONS:
        key, action_name = GESTURE_ACTIONS[fired]
        pyautogui.press(key)
        last_action = action_name
        last_action_time = time.time()

    # Top bar: gesture name + a bar that fills while the gesture is held
    label = GESTURE_LABELS.get(gesture, "-") if result.hand_landmarks else "No hand"
    cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    if gesture in GESTURE_ACTIONS:
        bar_width = int(width * trigger.progress())
        cv2.rectangle(frame, (0, 55), (bar_width, 60), (0, 255, 0), -1)

    # Show the last command for 1.5 seconds
    if time.time() - last_action_time < 1.5:
        cv2.putText(frame, last_action, (20, height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()
