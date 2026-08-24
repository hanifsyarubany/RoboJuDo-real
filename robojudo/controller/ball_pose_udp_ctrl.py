"""UDP transport for live ball perception -- third sibling of BallPoseRedisCtrl/BallPoseRos2Ctrl,
same {"kick_ball_pos_b", "kick_target_pos_b", "valid"} contract as both.

WHY THIS EXISTS, ON TOP OF THE OTHER TWO
BallPoseRedisCtrl needs Redis, which was tried and confirmed impossible to get running on a real
G1. BallPoseRos2Ctrl needs the consuming process (robojudo, Python 3.12/Humble) and the producing
process (a real detector, native Foxy/Python 3.8) to interoperate directly over the DDS wire --
unverified, and risky to bet a physical robot on without a real test (see
docs/g1_unified_loco_kick_deployment.md and docs/real_ball_perception_interface.md for the full
story). This controller needs neither: it's a plain UDP socket, stdlib-only (`socket`, `json`,
`threading`) on BOTH ends -- nothing to install, no daemon/server process beyond the two
application processes (the detector-side relay and this one) that were going to run anyway.

The Foxy-side counterpart is scripts/foxy_ros2_ball_bridge.py, which keeps ROS2 entirely inside
Foxy (same-distro rclpy, zero ABI risk -- this is the same reasoning that motivated the bridge
design in the first place, just with UDP instead of Redis as the hop across the Humble/Foxy
boundary) and sends this controller a UDP datagram per ROS2 message received.

WIRE FORMAT
One JSON object per UDP datagram, no framing needed (UDP is already message-oriented -- unlike TCP,
there's no stream to delimit), same shape as BallPoseRedisCtrl's Redis payload:
    {"kick_ball_pos_b": [x, y, z], "kick_target_pos_b": [x, y], "t": <unix seconds, float>}

UDP has no delivery guarantee and no connection state -- both correct fits here: a dropped datagram
just means the next one (arriving ~33ms later at 30Hz) supersedes it, and there's nothing to
reconnect after a network blip, unlike a broken TCP/Redis connection. staleness (via "t") already
existed to handle exactly this kind of gap, so nothing new is needed to tolerate loss."""

import json
import logging
import socket
import threading
import time
from collections import deque

import numpy as np

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import BallPoseUdpCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class BallPoseUdpCtrl(Controller):
    """UDP sibling of BallPoseRedisCtrl. See this module's docstring for the wire format."""

    cfg_ctrl: BallPoseUdpCtrlCfg

    def __init__(self, cfg_ctrl: BallPoseUdpCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        self.stale_after_s = cfg_ctrl.stale_after_s
        self.data_buffer = deque(maxlen=cfg_ctrl.buffer_size)
        self.last_data = None
        self._stop = False

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((cfg_ctrl.listen_host, cfg_ctrl.listen_port))
        self.sock.settimeout(0.5)  # periodic wakeup so the recv thread can notice self._stop

        self._recv_thread = threading.Thread(target=self._recv_worker, daemon=True)
        self._recv_thread.start()

        logger.info(
            f"[BallPoseUdpCtrl] Listening on {cfg_ctrl.listen_host}:{cfg_ctrl.listen_port} -- "
            f"waiting for first datagram (e.g. from scripts/foxy_ros2_ball_bridge.py)."
        )

        # Block for a first reading, same pattern as BallPoseRedisCtrl/BallPoseRos2Ctrl.
        waited = 0.0
        while self.last_data is None:
            self.get_data()
            time.sleep(0.01)
            waited += 0.01
            if int(waited) % 2 == 0 and abs(waited - round(waited)) < 0.005:
                logger.warning(
                    f"[BallPoseUdpCtrl] Still waiting for a first UDP datagram on port "
                    f"{cfg_ctrl.listen_port} ({waited:.0f}s) -- is the Foxy-side bridge running? "
                    f"e.g. python scripts/foxy_ros2_ball_bridge.py --dest-port {cfg_ctrl.listen_port}. "
                    f"Also double-check firewall/network reachability between the two hosts if "
                    f"they aren't the same machine."
                )
        logger.info("[BallPoseUdpCtrl] Initialized with first data.")

    def _recv_worker(self):
        while not self._stop:
            try:
                raw, _addr = self.sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                if self._stop:
                    return
                raise
            try:
                payload = json.loads(raw)
                reading = {
                    "kick_ball_pos_b": np.asarray(payload["kick_ball_pos_b"], dtype=np.float32),
                    "kick_target_pos_b": np.asarray(payload["kick_target_pos_b"], dtype=np.float32),
                    "t": float(payload["t"]),
                }
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"[BallPoseUdpCtrl] Malformed datagram: {e}")
                continue
            self.data_buffer.append(reading)

    def reset(self):
        self.data_buffer.clear()

    def get_data(self):
        """Pop oldest buffered reading -> last_data (non-blocking); valid=False if nothing has
        ever arrived, or the freshest thing we have is older than stale_after_s."""
        if len(self.data_buffer) > 0:
            self.last_data = self.data_buffer.popleft()

        data = self.last_data
        valid = data is not None and (time.time() - data["t"]) <= self.stale_after_s
        return {
            "kick_ball_pos_b": data["kick_ball_pos_b"] if data is not None else None,
            "kick_target_pos_b": data["kick_target_pos_b"] if data is not None else None,
            "valid": valid,
        }

    def shutdown(self):
        self._stop = True
        if self._recv_thread.is_alive():
            self._recv_thread.join(timeout=1.0)
        self.sock.close()
