"""Standalone tkinter compass window for the kick aim -- a real 2D GUI, not viewer markers.

Shows, for the CURRENTLY SELECTED kick skill: its nominal shot bearing, the trained
kick_aim_theta band around that nominal, and a live needle at nominal + the operator's current
aim (','/'.'/'0' keys). Sim-only in practice -- it is fed from run_pipeline_prepared.py, which only
constructs it when a viewer/sim run asks for it.

THREADING CONTRACT (the important part): the control loop must never wait on this window. update()
does nothing but rebind one dict -- no locks, no drawing, no I/O -- and the GUI thread polls that
dict on its own tkinter timer. A frozen, slow or closed window therefore cannot stall the control
loop, which matters here specifically: a blocking call on the loop thread is exactly what caused
the 45s/82s freezes diagnosed on 2026-09-15 (see robojudo/utils/logger.py's QueueHandler note).

Follows the tkinter-in-a-daemon-Thread pattern this project already uses for
controller/utils/motion_gui.py rather than introducing a second GUI toolkit.

ANGLE CONVENTION: degrees in the ROBOT's own frame -- 0 = straight forward, positive = the robot's
left, matching kick_aim_theta itself. Drawn with forward UP and the robot's left to the LEFT, i.e.
a top-down view of a robot facing up the screen.
"""

from __future__ import annotations

import logging
import math
from threading import Thread

logger = logging.getLogger(__name__)

_BG = "#12141a"
_RING = "#59627a"
_RING_TEXT = "#98a2b8"
_BAND = "#c87d14"
_NOMINAL = "#2fd46f"
_NEEDLE = "#ff4fc2"
_FG = "#e8ecf5"
_DIM = "#8a93a8"


class AimCompassGUI:
    """update() from the control loop; the window redraws itself on its own schedule."""

    def __init__(self, size: int = 340, scale: float = 1.0, refresh_hz: float = 20.0):
        """`scale` multiplies EVERY pixel dimension -- canvas, radii, line widths, font sizes -- so
        the whole widget is drawn at higher resolution rather than being a small window stretched
        by the window manager. Default 1.0 is the compact size; raise it on a HiDPI screen."""
        self._k = max(0.25, float(scale))
        self._size = int(round(size * self._k))
        self._interval_ms = max(1, int(1000.0 / max(refresh_hz, 1.0)))
        self._state: dict | None = None
        self._closed = False
        self.available = False  # flipped True once the window actually opens
        self._thread = Thread(target=self._run, name="AimCompassGUI", daemon=True)
        self._thread.start()

    # ---- called from the CONTROL LOOP: must stay trivial and non-blocking ----
    def update(self, state: dict | None):
        """Hand over the latest aim state (UnifiedLocoKickPolicy.get_aim_state()). One atomic
        rebind -- deliberately no lock: the GUI thread only ever reads the reference, so it either
        sees the previous dict or the new one, never a half-written one."""
        self._state = state

    def close(self):
        self._closed = True

    # ---- GUI thread ----
    def _run(self):
        try:
            import tkinter as tk
        except Exception as exc:  # tkinter not built into this interpreter
            logger.warning(f"[aim-compass] tkinter unavailable -- compass window disabled ({exc})")
            return
        try:
            root = tk.Tk()
        except Exception as exc:  # no DISPLAY (headless run over SSH, CI, container without X)
            logger.warning(
                f"[aim-compass] could not open a window -- compass disabled ({exc}). "
                "This needs a display; use the in-viewer ring (--show-aim-compass) instead."
            )
            return

        self._tk = tk
        root.title("Kick Aim Compass")
        root.configure(bg=_BG)
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._root = root

        # Deliberately NOT touching `tk scaling`: that scales POINT-sized fonts, and every font
        # below is already scaled explicitly by self._k. Doing both would square the factor. All
        # font sizes are given as NEGATIVE numbers, which Tk reads as pixels rather than points, so
        # the widget renders identically regardless of whatever scaling the desktop has set.
        s = self._size
        self._canvas = tk.Canvas(root, width=s, height=s + self._px(76), bg=_BG, highlightthickness=0)
        self._canvas.pack()
        self.available = True

        self._schedule()
        root.mainloop()
        self.available = False

    def _on_close(self):
        self._closed = True
        try:
            self._root.destroy()
        except Exception:
            pass

    def _schedule(self):
        if self._closed:
            self._on_close()
            return
        try:
            self._redraw()
        except Exception as exc:  # never let a draw bug kill the thread silently
            logger.warning(f"[aim-compass] redraw failed: {exc}")
        self._root.after(self._interval_ms, self._schedule)

    # ---- drawing ----
    def _px(self, v: float) -> float:
        """Scale a base-design pixel length by the DPI factor."""
        return v * self._k

    def _font(self, base_pt: int, bold: bool = False):
        """Font sized in PIXELS (negative = pixels in Tk), scaled by the DPI factor -- see _run()'s
        note on why this is done here rather than via `tk scaling`."""
        px = max(7, int(round(base_pt * 1.34 * self._k)))
        return ("TkDefaultFont", -px, "bold") if bold else ("TkDefaultFont", -px)

    def _xy(self, bearing_deg: float, radius: float):
        """Robot-frame bearing -> canvas point, forward UP and the robot's left to the LEFT."""
        a = math.radians(bearing_deg)
        return self._cx - radius * math.sin(a), self._cy - radius * math.cos(a)

    def _redraw(self):
        c = self._canvas
        c.delete("all")
        s = self._size
        self._cx, self._cy = s / 2.0, s / 2.0
        R = s / 2.0 - self._px(34)

        # ring + a tick every 30 deg, labelled with its signed robot-frame bearing
        c.create_oval(self._cx - R, self._cy - R, self._cx + R, self._cy + R,
                      outline=_RING, width=self._px(2))
        for k in range(12):
            b = k * 30.0
            b_signed = b if b <= 180 else b - 360
            cardinal = b_signed % 90 == 0
            x0, y0 = self._xy(b, R)
            x1, y1 = self._xy(b, R - self._px(11 if cardinal else 7))
            c.create_line(x0, y0, x1, y1, fill=_RING, width=self._px(2 if cardinal else 1))
            tx, ty = self._xy(b, R + self._px(15))
            c.create_text(tx, ty, text=f"{b_signed:+.0f}" if b_signed else "0", fill=_RING_TEXT,
                          font=self._font(8))

        st = self._state
        if st is None:
            c.create_text(self._cx, self._cy, text="no aim metadata\non this checkpoint",
                          fill=_DIM, justify="center", font=self._font(10))
            self._robot_marker()
            return

        nominal, max_deg, aim = st["nominal_deg"], st["max_deg"], st["aim_deg"]
        heading = st["heading_deg"]

        # trained band as a filled wedge. tkinter arcs measure CCW from east; our bearing measures
        # CCW from north (forward), so the offset is a straight +90.
        c.create_arc(self._cx - R, self._cy - R, self._cx + R, self._cy + R,
                     start=(nominal - max_deg) + 90.0, extent=2.0 * max_deg,
                     fill=_BAND, outline="", stipple="gray50")
        for edge in (nominal - max_deg, nominal + max_deg):
            ex, ey = self._xy(edge, R)
            c.create_line(self._cx, self._cy, ex, ey, fill=_BAND, width=self._px(1),
                          dash=(int(self._px(3)) or 1, int(self._px(3)) or 1))

        # nominal
        nx, ny = self._xy(nominal, R)
        c.create_line(self._cx, self._cy, nx, ny, fill=_NOMINAL, width=self._px(2))
        dot = self._px(5)
        c.create_oval(nx - dot, ny - dot, nx + dot, ny + dot, fill=_NOMINAL, outline="")

        # live needle
        hx, hy = self._xy(heading, R - self._px(4))
        c.create_line(self._cx, self._cy, hx, hy, fill=_NEEDLE, width=self._px(4),
                      arrow=self._tk.LAST,
                      arrowshape=(self._px(14), self._px(17), self._px(6)))

        self._robot_marker()

        # readout
        pad, y = self._px(12), s + self._px(10)
        row2, row3 = self._px(22), self._px(44)
        col2 = self._px(92)
        # Display only, 1-based -- st["skill_id"] itself stays 0-based everywhere else (it indexes
        # directly into the checkpoint's own per-skill arrays: _skill_ball_xy, _skill_target_xy,
        # _skill_kick_aim_max_deg, ...), this is purely what the operator reads on screen.
        c.create_text(pad, y, anchor="w", fill=_FG, font=self._font(11, bold=True),
                      text=f"skill {st['skill_id'] + 1}")
        c.create_text(col2, y, anchor="w", fill=_NOMINAL, font=self._font(10),
                      text=f"nominal {nominal:+.1f}°")
        c.create_text(pad, y + row2, anchor="w", fill=_NEEDLE, font=self._font(11, bold=True),
                      text=f"aim {aim:+.1f}°")
        c.create_text(col2, y + row2, anchor="w", fill=_BAND, font=self._font(10),
                      text=f"range ±{max_deg:.0f}°")
        c.create_text(pad, y + row3, anchor="w", fill=_FG, font=self._font(10),
                      text=f"kick heading {heading:+.1f}°")
        if not st.get("manual_enabled", True):
            c.create_text(self._size - pad, y + row3, anchor="e", fill=_DIM,
                          font=self._font(8), text="manual aim OFF")

    def _robot_marker(self):
        """Small triangle at the centre pointing up -- the robot, facing forward."""
        up, side, down = self._px(9), self._px(6), self._px(6)
        self._canvas.create_polygon(
            self._cx, self._cy - up, self._cx - side, self._cy + down, self._cx + side, self._cy + down,
            fill=_DIM, outline="",
        )
