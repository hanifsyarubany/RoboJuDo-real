"""Foxy-side ROS2 -> UDP bridge for live ball perception on the real G1.

RUN THIS UNDER THE ROBOT'S OWN NATIVE ROS2 FOXY PYTHON (e.g. after `source /opt/ros/foxy/
setup.bash`), NOT under `conda activate robojudo` -- this script deliberately imports nothing from
the robojudo package (no torch, no robojudo.pipeline) and needs NOTHING beyond the Python stdlib on
top of whatever Foxy already gives you (`socket`, `json` -- no pip install of anything at all). It
is NOT run on the same Python/Distro as robojudo itself.

WHY UDP, NOT REDIS
The original version of this bridge relayed to robojudo over Redis. That was tried on a real G1 and
confirmed impossible to get running there. Rather than chase a different specific technology that
might hit the same wall for the same unknown reason, this version removes every third-party
dependency from the Humble/Foxy boundary entirely: a UDP datagram needs no server process, no
daemon, nothing installed on either end beyond the two application processes (this bridge, and
robojudo's BallPoseUdpCtrl) that were going to run anyway.

WHY A BRIDGE AT ALL, INSTEAD OF ROBOJUDO'S BallPoseRos2Ctrl TALKING TO FOXY DIRECTLY
robojudo needs Python 3.12 (Humble's only numpy2-compatible rclpy build coexists with onnxruntime;
every Humble/py3.11 rclpy build in RoboStack forces numpy<2, which breaks onnxruntime outright --
confirmed by direct test, not assumption). The G1's onboard ROS2 is Foxy (Python 3.8). Whether a
Humble rclpy node and a Foxy rclpy node actually interoperate over the DDS/RTPS wire on the same
host is UNVERIFIED here: reconstructing a Foxy rclpy in an isolated sandbox to test it hit real,
unfixable C++ ABI breaks (spdlog/fmt version skew from RoboStack's Foxy channel being EOL and
unmaintained), so the cross-distro pub/sub test was never actually run.

Rather than bet a physical robot on an unverified DDS interop link, this script keeps Humble and
Foxy from ever having to talk to each other over DDS at all: it runs entirely INSIDE Foxy (so rclpy
talking to rclpy is same-distro, zero ABI risk) and hands off to robojudo over UDP instead.
robojudo's BallPoseUdpCtrl is a small, dependency-free listener built specifically for this.

TOPIC CONTRACT (matches robojudo/controller/ball_pose_ros2_ctrl.py's BallPoseRos2Ctrl exactly, so a
real detector can be written once and pointed at either consumer without caring which one is
actually listening):
  /ball_pose (geometry_msgs/PointStamped)   -- required. Robot heading frame (x-forward, z-up).
      No detection this tick = don't publish; do not publish a stale/fabricated position.
  /kick_aim  (geometry_msgs/Vector3Stamped) -- optional. Only .x/.y are read. A CONSTANT bounded
      bearing command (see dummy_ball_perception.py's --kick-aim-enabled docs) for an
      aim-enabled checkpoint, held for the whole kick attempt -- not a live per-tick value. If
      nothing is ever published here it stays [0, 0], the correct "no aim override" default.

Neither topic needs QoS RELIABLE -- this bridge subscribes BEST_EFFORT/VOLATILE (the most
permissive a subscriber can offer), matching BallPoseRos2Ctrl's own default, so it connects
regardless of whether your detector publishes BEST_EFFORT or RELIABLE.

STALENESS
BallPoseUdpCtrl's staleness check (default 0.5s) is judged against the "t" field this bridge sends.
To keep that honest end-to-end (a ball detector that silently stopped tracking should read as stale
in robojudo too, not as "still fresh because the bridge kept re-sending the last value"), "t" is
the ROS2 message's own header stamp when set, NOT this process's own wall-clock time at relay -- so
a frozen upstream publisher shows up as a growing age, exactly like a frozen
dummy_ball_perception.py would.

UDP has no delivery guarantee and no connection state -- both a correct fit here, not a compromise:
a dropped datagram just means the next one (arriving ~33ms later at 30Hz) supersedes it, and
there's nothing to "reconnect" after a network blip, unlike Redis or a TCP link. Nothing extra is
needed to tolerate loss -- the staleness check already existed to handle exactly this kind of gap.
"""

from __future__ import annotations

import argparse
import json
import logging
import socket

import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("foxy_ros2_ball_bridge")


def _stamp_to_sec(stamp) -> float | None:
    """Header stamp -> float seconds, or None if the publisher left it unset (all-zero) -- an
    unset stamp must NOT be read as 1970 (which would make every message look infinitely stale)."""
    sec, nanosec = int(stamp.sec), int(stamp.nanosec)
    if sec == 0 and nanosec == 0:
        return None
    return sec + nanosec * 1e-9


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ball-topic", default="/ball_pose", help="geometry_msgs/PointStamped, robot heading frame.")
    ap.add_argument("--aim-topic", default="/kick_aim", help="geometry_msgs/Vector3Stamped, optional.")
    ap.add_argument("--node-name", default="foxy_ros2_ball_bridge")
    ap.add_argument(
        "--dest-host", default="localhost",
        help="Host running run_pipeline_prepared.py --ball-source udp. If robojudo runs on a "
        "different machine than this bridge, this must be that machine's real address, not localhost.",
    )
    ap.add_argument(
        "--dest-port", type=int, default=7790,
        help="Must match BallPoseUdpCtrlCfg.listen_port / run_pipeline_prepared.py's --udp-listen-port.",
    )
    return ap.parse_args()


class Bridge(Node):
    def __init__(self, args: argparse.Namespace, sock: socket.socket, dest: tuple[str, int]):
        super().__init__(args.node_name)
        self.sock = sock
        self.dest = dest
        self.aim = [0.0, 0.0]

        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(PointStamped, args.ball_topic, self._on_ball, qos)
        self.create_subscription(Vector3Stamped, args.aim_topic, self._on_aim, qos)
        logger.info(
            f"Subscribed to '{args.ball_topic}' (PointStamped) and '{args.aim_topic}' (Vector3Stamped) "
            f"-- relaying to {dest[0]}:{dest[1]} over UDP."
        )

    def _on_aim(self, msg: Vector3Stamped):
        self.aim = [float(msg.vector.x), float(msg.vector.y)]

    def _on_ball(self, msg: PointStamped):
        t = _stamp_to_sec(msg.header.stamp)
        if t is None:
            # Unset stamp -- fall back to arrival time so staleness still measures something
            # meaningful ("time since we last heard anything"), just without the publisher's own
            # send time factored in.
            t = self.get_clock().now().nanoseconds * 1e-9

        payload = json.dumps(
            {
                "kick_ball_pos_b": [float(msg.point.x), float(msg.point.y), float(msg.point.z)],
                "kick_target_pos_b": self.aim,
                "t": t,
            }
        )
        # No try/except needed: unlike a Redis SET or a TCP send, a UDP sendto() on a connectionless
        # socket doesn't raise on "the other side isn't listening" -- it's fire-and-forget by design,
        # which is exactly the semantics wanted here (see this module's STALENESS section).
        self.sock.sendto(payload.encode("utf-8"), self.dest)


def main() -> int:
    args = parse_args()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.dest_host, args.dest_port)

    rclpy.init()
    node = Bridge(args, sock, dest)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
