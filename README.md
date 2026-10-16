# Hand Gesture Video Control

Control any video player (YouTube, VLC, Netflix, Spotify, ...) with hand gestures through your webcam. No mouse, no keyboard.

Built with **Python**, **OpenCV** and **Google MediaPipe**.

<!-- Demo GIF here: ![Demo](demo.gif) -->

## Gestures

| Gesture | Action |
|---|---|
| ✋ Open palm | Play / Pause |
| 👍 Thumbs up | Mute / Unmute |
| 👈 Point left | Rewind (hold to keep rewinding) |
| 👉 Point right | Fast forward (hold to keep going) |
| ☝️ Index up + thumb | Volume mode: spread fingers for louder, pinch for quieter |
| ✊ Fist | Rest (does nothing) |

Keyboard: `H` toggle help · `P` pause gesture control · `Q` quit

## How it works

```
Webcam frame ─► MediaPipe Hand Landmarker ─► 21 hand landmarks
                                                  │
                     ┌────────────────────────────┘
                     ▼
         Which fingers are open?  (geometry on landmarks)
                     │
                     ▼
         Which gesture is it?     (rules on finger states + index direction)
                     │
                     ▼
         Should it fire now?      (stability frames, cooldown, repeat timing)
                     │
                     ▼
         Press a key              (media keys / arrow keys via PyAutoGUI)
```

**Finger state.** A finger is open when its tip is farther from the wrist than its middle joint. Using distances instead of "tip above joint" keeps it working when the hand is tilted or rotated. The thumb folds across the palm, so it is measured against the pinky base instead.

**Avoiding false triggers.** Hands pass through other shapes while changing gestures, so a gesture must be held for several consecutive frames before it fires, fires only once per hold, and has a cooldown. Seeking uses a separate trigger that repeats at a fixed rate while the gesture is held.

**Scale-independent volume.** The thumb–index distance is divided by the palm size, so the same gesture gives the same value whether the hand is near or far from the camera.

## Setup

Requires Python 3.10+ and a webcam.

```bash
pip install -r requirements.txt
python main.py
```

The hand model (~8 MB) downloads automatically on first run.

> Play/pause, mute and volume use media keys, so they work no matter which window is focused. Seeking uses arrow keys, so click on the video player window first.

## Project structure

```
main.py           App loop: camera → gesture → action → UI
hand_tracker.py   MediaPipe wrapper, returns 21 landmarks in pixels
gestures.py       Finger states and gesture recognition
triggers.py       When to fire: stability, cooldown, repeat, throttle
ui.py             On-screen overlay (hand, top bar, volume meter, help)
config.py         All thresholds and gesture→key mappings
steps/            The project built step by step, from opening the camera to full control
```

## Customizing

Everything is in [config.py](config.py): change which key each gesture presses, how long a gesture must be held, the volume thresholds, or the camera index.
