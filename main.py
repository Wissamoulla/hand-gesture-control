"""Hand Gesture Video Control: control any video player with hand gestures."""

import time

import cv2
import pyautogui

import config
import ui
from gestures import get_open_fingers, pinch_ratio, recognize_gesture
from hand_tracker import HandTracker
from triggers import GestureTrigger, RepeatTrigger, Throttle

ACTION_DISPLAY_SECONDS = 1.0


def main():
    tracker = HandTracker()
    trigger = GestureTrigger(config.STABLE_FRAMES, config.COOLDOWN_SECONDS)
    seek_trigger = RepeatTrigger(config.SEEK_STABLE_FRAMES, config.SEEK_REPEAT_SECONDS)
    volume_throttle = Throttle(config.VOLUME_STEP_SECONDS)

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open the camera")
        return

    paused = False
    show_help = True
    last_action = ""
    last_action_time = 0.0
    fps = 0.0
    prev_frame_time = time.time()

    def run(key, action_name):
        nonlocal last_action, last_action_time
        pyautogui.press(key)
        last_action = action_name
        last_action_time = time.time()

    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read a frame")
            break
        frame = cv2.flip(frame, 1)  # mirror view

        # Smoothed frames-per-second
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - prev_frame_time, 1e-6))
        prev_frame_time = now

        points = tracker.find_hand(frame)
        gesture = None

        if points:
            fingers = get_open_fingers(points)
            gesture = recognize_gesture(fingers, points)
            ui.draw_hand(frame, points, fingers)

            if gesture == "VOLUME":
                ratio = pinch_ratio(points)
                ui.draw_volume_meter(frame, points, ratio)
                if not paused and volume_throttle.ready():
                    if ratio > config.VOLUME_UP_RATIO:
                        run("volumeup", "Volume +")
                    elif ratio < config.VOLUME_DOWN_RATIO:
                        run("volumedown", "Volume -")

        # While paused, triggers still see "no gesture" so nothing fires
        active_gesture = None if paused else gesture

        fired = trigger.update(active_gesture)
        if fired in config.GESTURE_ACTIONS:
            run(*config.GESTURE_ACTIONS[fired])

        seek = seek_trigger.update(active_gesture if active_gesture in config.SEEK_ACTIONS else None)
        if seek:
            run(*config.SEEK_ACTIONS[seek])

        # --- Draw the interface ---
        label = config.GESTURE_LABELS.get(gesture, "-") if points else "No hand"
        progress = trigger.progress() if active_gesture in config.GESTURE_ACTIONS else 0.0
        ui.draw_top_bar(frame, label, progress, fps, paused)
        if show_help:
            ui.draw_help(frame)
        if time.time() - last_action_time < ACTION_DISPLAY_SECONDS:
            ui.draw_action(frame, last_action)

        cv2.imshow("Hand Gesture Video Control", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("p"):
            paused = not paused
        if key == ord("h"):
            show_help = not show_help

    tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
