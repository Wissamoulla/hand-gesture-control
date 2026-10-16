"""Everything drawn on top of the camera image."""

import cv2

import config

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colors are BGR in OpenCV
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 220, 0)
RED = (0, 0, 230)
YELLOW = (0, 230, 255)
PURPLE = (255, 0, 255)
GRAY = (160, 160, 160)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # pinky + palm
]
FINGER_TIPS = [4, 8, 12, 16, 20]


def draw_hand(frame, points, fingers):
    """Skeleton in white, finger tips green (open) or red (closed)."""
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], WHITE, 2)
    for tip, is_open in zip(FINGER_TIPS, fingers):
        cv2.circle(frame, points[tip], 9, GREEN if is_open else RED, -1)


def draw_top_bar(frame, label, progress, fps, paused):
    """Gesture name, a hold-progress bar, FPS and pause state."""
    width = frame.shape[1]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 60), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, label, (20, 40), FONT, 1.0, YELLOW, 2)
    cv2.putText(frame, f"{fps:.0f} FPS", (width - 110, 40), FONT, 0.7, GRAY, 2)

    if paused:
        cv2.putText(frame, "PAUSED", (width // 2 - 60, 40), FONT, 1.0, RED, 2)
    if progress > 0:
        cv2.rectangle(frame, (0, 56), (int(width * progress), 60), GREEN, -1)


def draw_volume_meter(frame, points, ratio):
    """Line between thumb and index, plus a vertical meter with the two thresholds."""
    height, width, _ = frame.shape
    cv2.line(frame, points[4], points[8], PURPLE, 3)

    top, bottom = 100, height - 100
    left, right = width - 50, width - 20

    def y_for(value):
        return int(bottom - min(value, 1.0) * (bottom - top))

    cv2.rectangle(frame, (left, top), (right, bottom), WHITE, 2)
    cv2.rectangle(frame, (left, y_for(ratio)), (right, bottom), PURPLE, -1)
    cv2.line(frame, (left - 10, y_for(config.VOLUME_UP_RATIO)), (right + 10, y_for(config.VOLUME_UP_RATIO)), GREEN, 2)
    cv2.line(frame, (left - 10, y_for(config.VOLUME_DOWN_RATIO)), (right + 10, y_for(config.VOLUME_DOWN_RATIO)), RED, 2)


def draw_action(frame, text):
    """The last command, at the bottom of the screen."""
    height = frame.shape[0]
    cv2.putText(frame, text, (20, height - 30), FONT, 1.2, BLACK, 6)
    cv2.putText(frame, text, (20, height - 30), FONT, 1.2, GREEN, 2)


def draw_help(frame):
    """Semi-transparent panel listing the gestures."""
    line_height = 20
    panel_height = 35 + line_height * len(config.HELP_LINES)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 70), (290, 70 + panel_height), BLACK, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Two columns, since the font is not fixed-width
    for i, (gesture, action) in enumerate(config.HELP_LINES):
        y = 90 + i * line_height
        cv2.putText(frame, gesture, (20, y), FONT, 0.45, YELLOW, 1, cv2.LINE_AA)
        cv2.putText(frame, action, (115, y), FONT, 0.45, WHITE, 1, cv2.LINE_AA)

    keys_y = 90 + len(config.HELP_LINES) * line_height + 5
    cv2.putText(frame, config.HELP_KEYS, (20, keys_y), FONT, 0.45, GRAY, 1, cv2.LINE_AA)
