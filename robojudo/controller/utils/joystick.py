import logging
import math
import os
import struct
import time
from queue import Empty, Queue
from threading import Thread

import numpy as np

logger = logging.getLogger(__name__)

# Sanity bound for the real remote's raw axis floats -- generously wider than the expected [-1, 1]
# joystick range (to tolerate normal out-of-calibration sticks), but tight enough to catch garbage
# from a corrupted wireless packet (RF interference near motors is a real, sim-has-no-equivalent-of
# condition on this link -- struct.unpack on a bad byte pattern can produce NaN/Inf or huge floats).
_REMOTE_AXIS_SANITY_BOUND = 1.5
_REMOTE_AXIS_WARN_INTERVAL_S = 1.0  # rate-limit the invalid-reading warning, don't spam per-tick

_REMOTE_DIAG_REPORT_INTERVAL_S = 5.0  # how often to log the parse()-rate / repeated-buffer summary

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"


# TODO: axis post processing
class JoystickThread(Thread):
    def __init__(self, state_queue: Queue, event_queue: Queue):
        super().__init__(name="JoystickThread", daemon=True)
        self.state_queue = state_queue
        self.event_queue = event_queue

        self.config = self._init_config()

        self.running = True

    # fmt: off
    def _init_config(self):
        config = {
            "button_map": {
                0: "A", 1: "B", 2: "X", 3: "Y",
                4: "LB", 5: "RB", 6: "Back", 7: "Start",
                8: "Xbox", 9: "L", 10: "R",
            },
            "axis_config": {
                "axis_map": {
                    "LeftX": 0,
                    "LeftY": 1,
                    "RightX": 3,
                    "RightY": 4,
                    "LT": 2,
                    "RT": 5
                },
                "axis_range": {
                    "LT": [0, 1],
                    "RT": [0, 1]
                },
                "invert": ["LeftY", "RightY"],
            },
            "dpad_config": {
                "as_button_event": True,  # map dpad to button events
                "dpad_map": {
                    "Up": (1, 1),
                    "Right": (0, 1),
                    "Down": (1, -1),
                    "Left": (0, -1),
                }

            }
        }

        # if windows
        if os.name == 'nt':  # Windows
            config["button_map"].update({
                8: "L", 9: "R",
            })
            del config["button_map"][10]
            config["axis_config"]["axis_map"].update({
                "RightX": 2,
                "RightY": 3,
                "LT": 4,
            })
        return config
    # fmt: on

    @staticmethod
    def normalize_axis(axis_range, name, value):
        target_range = axis_range.get(name)
        if target_range:
            min_val, max_val = -1.0, 1.0  # SDL default range
            min_target, max_target = target_range
            value = (value - min_val) / (max_val - min_val) * (max_target - min_target) + min_target
        return round(value, 3)

    def run(self):
        import pygame

        pygame.init()
        if pygame.joystick.get_count() == 0:
            self.running = False
            logger.error("No joystick connected. Try to fix with: export SDL_JOYSTICK_DEVICE=/dev/input/js0")
            # raise RuntimeError("No joystick connected, try to fix with: export SDL_JOYSTICK_DEVICE=/dev/input/js0")
            return

        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        name = joystick.get_name().lower()
        logger.info(f"[Joystick] Initialized: {name}")
        logger.info(
            f"[Joystick] Buttons: {joystick.get_numbuttons()}, \
                Axes: {joystick.get_numaxes()}, \
                Hats: {joystick.get_numhats()}"
        )

        button_map = self.config.get("button_map", {})
        axis_config = self.config.get("axis_config", {})

        dpad_config = self.config.get("dpad_config", {})
        dpad_as_button = dpad_config.get("as_button_event", True)
        dpad_state = {key: False for key in dpad_config.get("dpad_map", {}).keys()}

        axis_map = axis_config.get("axis_map", {})
        axis_range = axis_config.get("axis_range", {})
        invert = set(axis_config.get("invert", []))

        clock = pygame.time.Clock()
        last_state_time = time.time()
        state_interval = 1.0 / 100  # 100Hz

        while self.running:
            pygame.event.pump()
            now = time.time()

            # Poll events for buttons and DPad
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN or event.type == pygame.JOYBUTTONUP:
                    btn_index = event.button
                    btn_name = button_map.get(btn_index, f"Button_{btn_index}")
                    self.event_queue.put(
                        {
                            "type": "button",
                            "name": btn_name,
                            "pressed": event.type == pygame.JOYBUTTONDOWN,
                            "timestamp": now,
                        }
                    )

                elif event.type == pygame.JOYHATMOTION:
                    if dpad_as_button:
                        dpad_state_new = {
                            name: event.value[axis] == direction
                            for name, (axis, direction) in dpad_config.get("dpad_map", {}).items()
                        }
                        for name, pressed in dpad_state_new.items():
                            if pressed != dpad_state[name]:
                                dpad_state[name] = pressed
                                self.event_queue.put(
                                    {
                                        "type": "button",
                                        "name": name,
                                        "pressed": pressed,
                                        "timestamp": now,
                                    }
                                )
                    else:
                        self.event_queue.put(
                            {
                                "type": "dpad",
                                "value": event.value,
                                "timestamp": now,
                            }
                        )

            # Axes update at fixed rate
            if now - last_state_time >= state_interval:
                axes_state = {}
                for name, index in axis_map.items():
                    val = joystick.get_axis(index)
                    if name in invert:
                        val = -val
                    val = self.normalize_axis(axis_range, name, val)
                    axes_state[name] = val

                while self.state_queue.full():
                    self.state_queue.get()
                self.state_queue.put(
                    {
                        "type": "axes",
                        "axes": axes_state,
                        "timestamp": now,
                    }
                )
                last_state_time = now

            clock.tick(500)  # avoid busy loop


# Modified From unitree_sdk2_python
class unitreeRemoteController:
    def __init__(self, state_queue, event_queue):
        self.state_queue = state_queue
        self.event_queue = event_queue

        # button
        self.button_map = [
            "R1",
            "L1",
            "Start",
            "Select",
            "R2",
            "L2",
            "F1",
            "F2",
            "A",
            "B",
            "X",
            "Y",
            "Up",
            "Right",
            "Down",
            "Left",
        ]
        self.last_button_state = np.zeros((16), dtype=bool)

        # last-known-good axis readings -- a rejected (NaN/Inf/out-of-range) tick holds the previous
        # value rather than snapping to 0.0, matching what a real analog stick physically does during
        # a brief glitch (freezes) instead of injecting an unintended "release to center" transient.
        self.last_axes = {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0, "RightY": 0.0}
        self._last_axis_warn_t = 0.0

        # --- update-rate / staleness diagnostic (log-only, changes no behavior) ---
        # Answers "how often does parse() actually get called, and is it ever handed the exact
        # same bytes twice in a row" -- unlike JoystickCtrl's sim path (a dedicated thread polling
        # the SDL gamepad at up to 500Hz, decoupled from the 50Hz control loop), UnitreeCtrl skips
        # that thread entirely and wires this parse() call directly into env.update(), i.e. it only
        # ever runs at whatever cadence the control loop itself actually achieves -- see this
        # module's own real-vs-sim rate discussion. A byte-identical consecutive buffer is
        # consistent with EITHER the stick genuinely not having moved (same physical position ->
        # same ADC reading -> same encoded bytes) OR the underlying SDK feed not refreshing every
        # tick -- this cannot distinguish the two on its own, so the periodic summary reports the
        # raw numbers and leaves interpretation to whether the stick was actually being moved at
        # the time.
        self._diag_last_call_t: float | None = None
        self._diag_last_raw: bytes | None = None
        self._diag_samples = 0  # number of call-to-call gaps observed in the current window
        self._diag_repeat_count = 0
        self._diag_repeat_streak = 0
        self._diag_max_repeat_streak = 0
        self._diag_dt_sum = 0.0
        self._diag_dt_max = 0.0
        self._diag_last_report_t = time.time()

    def _update_rate_diagnostic(self, remote_data) -> None:
        now = time.time()
        raw = bytes(remote_data)  # snapshot -- defensive in case the caller reuses its buffer

        if self._diag_last_call_t is not None:
            dt = now - self._diag_last_call_t
            self._diag_dt_sum += dt
            self._diag_dt_max = max(self._diag_dt_max, dt)
            self._diag_samples += 1
            if raw == self._diag_last_raw:
                self._diag_repeat_count += 1
                self._diag_repeat_streak += 1
                self._diag_max_repeat_streak = max(self._diag_max_repeat_streak, self._diag_repeat_streak)
            else:
                self._diag_repeat_streak = 0

        self._diag_last_call_t = now
        self._diag_last_raw = raw

        if now - self._diag_last_report_t >= _REMOTE_DIAG_REPORT_INTERVAL_S and self._diag_samples > 0:
            mean_hz = self._diag_samples / self._diag_dt_sum if self._diag_dt_sum > 0 else float("nan")
            repeat_pct = 100.0 * self._diag_repeat_count / self._diag_samples
            logger.info(
                f"[unitreeRemoteController] rate diag (last {_REMOTE_DIAG_REPORT_INTERVAL_S:.0f}s): "
                f"parse() called {self._diag_samples + 1} times, mean {mean_hz:.1f} Hz, worst gap "
                f"{self._diag_dt_max * 1000:.0f} ms, {repeat_pct:.0f}% identical-to-previous-buffer "
                f"(longest streak {self._diag_max_repeat_streak} calls). A high repeat% while the "
                "stick is untouched is expected. A high repeat% or long streak WHILE actively moving "
                "the stick would mean the underlying SDK feed isn't refreshing every tick -- not just "
                "the control loop running slow."
            )
            self._diag_last_report_t = now
            self._diag_samples = 0
            self._diag_repeat_count = 0
            self._diag_max_repeat_streak = 0
            self._diag_dt_sum = 0.0
            self._diag_dt_max = 0.0

    def _sanitize_axis(self, name: str, value: float) -> float:
        if math.isfinite(value) and abs(value) <= _REMOTE_AXIS_SANITY_BOUND:
            self.last_axes[name] = value
            return value
        now = time.time()
        if now - self._last_axis_warn_t >= _REMOTE_AXIS_WARN_INTERVAL_S:
            self._last_axis_warn_t = now
            logger.warning(
                f"[unitreeRemoteController] rejected {name}={value!r} (NaN/Inf or |value| > "
                f"{_REMOTE_AXIS_SANITY_BOUND}) -- likely a corrupted wireless packet; holding last "
                f"good value {self.last_axes[name]!r} instead"
            )
        return self.last_axes[name]

    def parse(self, remoteData):
        self._update_rate_diagnostic(remoteData)

        now = time.time()
        # button
        keys = struct.unpack("H", remoteData[2:4])[0]
        button = [((keys & (1 << i)) >> i) for i in range(16)]
        button_state = np.array(button, dtype=bool)

        # Check for button state changes
        changed = button_state != self.last_button_state
        for i in range(16):
            if changed[i]:
                self.event_queue.put(
                    {
                        "type": "button",
                        "name": self.button_map[i],
                        "pressed": bool(button_state[i]),
                        "timestamp": now,
                    }
                )
        self.last_button_state = button_state.copy()

        # axis -- unpacked from the raw wireless payload with no transport-level integrity check
        # available to us (see _sanitize_axis's comment), so every value is validated before use.
        lx_offset = 4
        LeftX = self._sanitize_axis("LeftX", struct.unpack("<f", remoteData[lx_offset : lx_offset + 4])[0])
        rx_offset = 8
        RightX = self._sanitize_axis("RightX", struct.unpack("<f", remoteData[rx_offset : rx_offset + 4])[0])
        ry_offset = 12
        RightY = self._sanitize_axis("RightY", struct.unpack("<f", remoteData[ry_offset : ry_offset + 4])[0])
        # L2_offset = 16
        # L2 = struct.unpack('<f', remoteData[L2_offset:L2_offset + 4])[0] # Placeholder，unused
        ly_offset = 20
        LeftY = self._sanitize_axis("LeftY", struct.unpack("<f", remoteData[ly_offset : ly_offset + 4])[0])

        while self.state_queue.full():
            self.state_queue.get()
        self.state_queue.put(
            {
                "type": "axes",
                "axes": {
                    "LeftX": LeftX,
                    "LeftY": LeftY,
                    "RightX": RightX,
                    "RightY": RightY,
                },
                "timestamp": now,
            }
        )


if __name__ == "__main__":
    state_queue = Queue(maxsize=10)
    event_queue = Queue(maxsize=100)
    js_thread = JoystickThread(state_queue, event_queue)
    js_thread.start()

    print("Press joystick buttons (Ctrl+C to exit)...")
    try:
        while True:
            try:
                state = state_queue.get(timeout=1.0)
                print("State:", state)
            except Empty:
                pass

            while not event_queue.empty():
                try:
                    event = event_queue.get_nowait()
                    print("Event:", event)
                except Empty:
                    break
    except KeyboardInterrupt:
        print("Exiting...")
        js_thread.running = False
        js_thread.join()
