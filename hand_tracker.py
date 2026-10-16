"""Wraps the MediaPipe hand landmark model."""

import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

import config


def ensure_model():
    """Download the model file the first time the app runs."""
    if config.MODEL_PATH.exists():
        return
    print("Downloading hand model (about 8 MB)...")
    urllib.request.urlretrieve(config.MODEL_URL, config.MODEL_PATH)
    print("Done.")


class HandTracker:
    def __init__(self):
        ensure_model()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.start_time = time.time()

    def find_hand(self, frame):
        """Return the 21 landmarks of the hand as pixel (x, y) points, or None."""
        height, width, _ = frame.shape

        # MediaPipe expects RGB, OpenCV gives BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode needs an increasing timestamp in milliseconds
        timestamp_ms = int((time.time() - self.start_time) * 1000)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks:
            return None
        hand = result.hand_landmarks[0]
        return [(int(lm.x * width), int(lm.y * height)) for lm in hand]

    def close(self):
        self.detector.close()
