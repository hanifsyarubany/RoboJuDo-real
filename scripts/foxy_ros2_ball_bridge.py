"""Foxy-side ROS2 -> Redis bridge for live ball perception on the real G1.

RUN THIS UNDER THE ROBOT'S OWN NATIVE ROS2 FOXY PYTHON (e.g. after `source /opt/ros/foxy/
setup.bash`), NOT under `conda activate robojudo` -- this script deliberately does not import
anything from the robojudo package (no torch, no robojudo.pipeline) so it stays installable with
just `pip install redis` on top of whatever Foxy already gives you. It is NOT run on the same
Python/Distro as robojudo itself.

WHY THIS EXISTS, INSTEAD OF ROBOJUDO'S BallPoseRos2Ctrl TALKING TO FOXY DIRECTLY
robojudo needs Python 3.12 (Humble's only numpy2-compatible rclpy build coexists with onnxruntime;
every Humble/py3.11 rclpy build in RoboStack forces numpy<2, which breaks onnxruntime outright --
confirmed by direct test, not assumption). The G1's onboard ROS2 is Foxy (Python 3.8). Whether a
Humble rclpy node and a Foxy rclpy node actually interoperate over the DDS/RTPS wire on the same
host is UNVERIFIED here: reconstructing a Foxy rclpy in this sandbox to test it hit real, unfixable
C++ ABI breaks (spdlog/fmt version skew from RoboStack's Foxy channel being EOL and unmaintained),
so the cross-distro pub/sub test was never actually run.

Rather than bet a physical robot on an unverified DDS interop link, this script keeps Humble and
Foxy from ever having to talk to each other over DDS at all: it runs entirely INSIDE Foxy (so
rclpy talking to rclpy is same-distro, zero ABI risk) and hands off to robojudo over Redis instead
-- the same, already-verified transport dummy_ball_perception.py uses for sim testing. robojudo's
BallPoseRedisCtrl is completely unmodified and unaware this bridge exists.

TOPIC CONTRACT (matches robojudo/controller/ball_pose_ros2_ctrl.py's BallPoseRos2Ctrl exactly, so
a real detector can be written once and pointed at either consumer without caring which one is
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
Redis staleness (BallPoseRedisCtrl.stale_after_s, default 0.5s) is judged against the "t" field
this bridge writes. To keep that honest end-to-end (a ball detector that silently stopped tracking
should read as stale in robojudo too, not as "still fresh because the bridge kept republishing the
last value"), "t" is the ROS2 message's own header stamp when set, NOT this process's own
wall-clock time at relay -- so a frozen upstream publisher shows up as a growing age, exactly like
a frozen dummy_ball_perception.py would.
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import redis
from redis.exceptions import RedisError

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
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--redis-db", type=int, default=0)
    ap.add_argument("--redis-key", default="ball_pose_g1", help="Must match BallPoseRedisCtrlCfg.redis_key.")
    return ap.parse_args()


class Bridge(Node):
    def __init__(self, args: argparse.Namespace, redis_client: redis.Redis):
        super().__init__(args.node_name)
        self.redis_client = redis_client
        self.redis_key = args.redis_key
        self.aim = [0.0, 0.0]
        self._redis_failing = False  # throttle: warn once per failure streak, not every message

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
            f"-- relaying to redis key '{self.redis_key}' on {args.redis_host}:{args.redis_port}/{args.redis_db}."
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
        try:
            self.redis_client.set(self.redis_key, payload)
            self._redis_failing = False
        except RedisError as e:
            if not self._redis_failing:
                logger.warning(f"Lost redis connection ({e}), will keep retrying silently until it recovers...")
                self._redis_failing = True


def connect_redis(host: str, port: int, db: int) -> redis.Redis:
    while True:
        try:
            client = redis.Redis(host=host, port=port, db=db, socket_timeout=1, socket_connect_timeout=1)
            client.ping()
            logger.info(f"Connected to redis at {host}:{port}/{db}")
            return client
        except Exception as e:
            logger.error(f"Redis connect failed ({e}), retrying...")
            time.sleep(0.5)


def main() -> int:
    args = parse_args()
    redis_client = connect_redis(args.redis_host, args.redis_port, args.redis_db)

    rclpy.init()
    node = Bridge(args, redis_client)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
