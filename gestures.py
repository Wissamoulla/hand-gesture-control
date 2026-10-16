"""Turns 21 hand landmarks into a gesture name."""

import math

import config

# (tip, middle joint) for the four long fingers
FINGER_JOINTS = [(8, 6), (12, 10), (16, 14), (20, 18)]


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


def recognize_gesture(fingers, points):
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
        # Only the index is open: its direction decides the gesture
        dx = points[8][0] - points[5][0]
        dy = points[8][1] - points[5][1]
        if abs(dx) > abs(dy) * config.POINT_SIDEWAYS_RATIO:
            return "POINT_RIGHT" if dx > 0 else "POINT_LEFT"
        return "VOLUME"
    return None


def pinch_ratio(points):
    """Thumb-index distance divided by palm size, so it ignores camera distance."""
    palm_size = distance(points[0], points[9])
    return distance(points[4], points[8]) / palm_size
