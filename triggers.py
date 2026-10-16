"""Decide *when* a gesture should run a command, to avoid accidental triggers."""

import time


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


class RepeatTrigger:
    """Fires once a gesture is steady, then keeps firing while it is held."""

    def __init__(self, stable_frames, repeat_seconds):
        self.stable_frames = stable_frames
        self.repeat_seconds = repeat_seconds
        self.current = None
        self.count = 0
        self.last_fire_time = 0.0

    def update(self, gesture):
        """Call once per frame. Returns the gesture when it should fire, else None."""
        if gesture == self.current:
            self.count += 1
        else:
            self.current = gesture
            self.count = 1
            self.last_fire_time = 0.0  # a new hold fires as soon as it is stable

        if gesture is None or self.count < self.stable_frames:
            return None
        if time.time() - self.last_fire_time >= self.repeat_seconds:
            self.last_fire_time = time.time()
            return gesture
        return None


class Throttle:
    """Allows an action at most once every `interval` seconds."""

    def __init__(self, interval):
        self.interval = interval
        self.last_time = 0.0

    def ready(self):
        if time.time() - self.last_time >= self.interval:
            self.last_time = time.time()
            return True
        return False
