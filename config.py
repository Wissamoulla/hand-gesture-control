"""All settings in one place. Tweak these to fit your hand, camera and player."""

from pathlib import Path

# --- Model & camera ---
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
CAMERA_INDEX = 0                  # 0 = built-in camera, 1 = external camera
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# --- Still gestures (play/pause, mute): fire once per hold ---
STABLE_FRAMES = 8        # frames in a row a gesture must hold
COOLDOWN_SECONDS = 1.0   # minimum time between two commands

# --- Pointing (seek): repeat while held ---
SEEK_STABLE_FRAMES = 5
SEEK_REPEAT_SECONDS = 0.5
POINT_SIDEWAYS_RATIO = 1.5  # how horizontal the index must be to count as pointing

# --- Volume: pinch distance relative to palm size ---
VOLUME_UP_RATIO = 0.7
VOLUME_DOWN_RATIO = 0.25
VOLUME_STEP_SECONDS = 0.1

# --- Gesture -> action mappings ---
GESTURE_LABELS = {
    "OPEN_PALM": "Open Palm",
    "FIST": "Fist",
    "THUMBS_UP": "Thumbs Up",
    "PEACE": "Peace (V)",
    "VOLUME": "Volume Mode",
    "POINT_LEFT": "Point Left",
    "POINT_RIGHT": "Point Right",
}

# Media keys work without focusing the player (YouTube, VLC, Spotify, ...)
GESTURE_ACTIONS = {
    "OPEN_PALM": ("playpause", "Play / Pause"),
    "THUMBS_UP": ("volumemute", "Mute / Unmute"),
}

# Arrow keys go to the focused window, so click the player first
SEEK_ACTIONS = {
    "POINT_LEFT": ("left", "<< Back"),
    "POINT_RIGHT": ("right", ">> Forward"),
}

# Shown in the help panel (H key): (gesture, action)
HELP_LINES = [
    ("Open palm", "Play / Pause"),
    ("Thumbs up", "Mute"),
    ("Point left", "Back (hold)"),
    ("Point right", "Forward (hold)"),
    ("Index up", "Volume: pinch / spread"),
    ("Fist", "Rest"),
]
HELP_KEYS = "H help   P pause   Q quit"
