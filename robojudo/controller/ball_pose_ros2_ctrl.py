"""ROS2 transport for the live ball/aim readings that BallPoseRedisCtrl carries over Redis.

Same contract, same consumer: get_data() returns
``{"kick_ball_pos_b": (3,), "kick_target_pos_b": (2,), "valid": bool}``, and
UnifiedLocoKickPolicy._get_live_ball_obs falls back to zeros whenever ``valid`` is False -- exactly
as it does for the Redis controller. Pick the transport with run_pipeline_prepared.py's
``--ball-source {redis,ros2}``.

WHY TWO TOPICS, NOT ONE
The Redis payload bundles the ball reading and the kick target into one JSON blob because a
key/value store gains nothing from splitting them. Over ROS2 they are better modelled as what they
actually are -- two signals from two different producers:

  ball_topic (geometry_msgs/PointStamped)  -- PERCEPTION. dummy_ball_perception.py in sim; the real
      onboard detector on the robot. This is the signal that must keep flowing and the one that can
      go stale (detector loses track), so it -- and only it -- drives ``valid``.
  aim_topic (geometry_msgs/Vector3Stamped) -- COMMAND, from the task/operator layer, not perception.
      For a kick_aim_enabled checkpoint it is a CONSTANT bounded bearing command held for the whole
      attempt (see dummy_ball_perception.py's module docstring), so it deliberately does NOT expire:
      the last value received stays in effect. If nothing was ever published it stays zero, which is
      also the correct "no aim" default.

Both are already in the robot's heading frame (x-forward, z-up) -- this controller does no frame
math at all, exactly like BallPoseRedisCtrl.

QoS
Subscriptions default to BEST_EFFORT/VOLATILE, the most permissive combination a subscriber can
offer: it connects to publishers that are RELIABLE or BEST_EFFORT, VOLATILE or TRANSIENT_LOCAL. A
stricter subscriber silently fails to connect to a laxer publisher -- no error, no data -- which is
the single most common way a ROS2 link "just doesn't work", so the default is the one that always
matches. Set ``qos_reliable=True`` only if your publisher is RELIABLE and you need delivery
guarantees.

There is no polling stage here: readings arrive on the executor thread as they are published.
(Contrast BallPoseRedisCtrl, whose worker re-reads the key every 10 ms into a maxlen=5 FIFO that
get_data() drains from the OLDEST end, so a value can sit behind several duplicate reads.)
"""

import logging
import threading
import time

import numpy as np

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import BallPoseRos2CtrlCfg

logger = logging.getLogger(__name__)


def _stamp_to_sec(stamp) -> float | None:
    """Header stamp -> float seconds, or None if the publisher left it unset (all-zero).

    An unset stamp is common in hand-rolled publishers and must NOT be read as "1970", which would
    make every message look infinitely stale and silently zero the ball observation forever.
    """
    sec = int(stamp.sec)
    nanosec = int(stamp.nanosec)
    if sec == 0 and nanosec == 0:
        return None
    return sec + nanosec * 1e-9


@ctrl_registry.register
class BallPoseRos2Ctrl(Controller):
    """ROS2 sibling of BallPoseRedisCtrl. See the module docstring for the topic/QoS contract."""

    cfg_ctrl: BallPoseRos2CtrlCfg

    def __init__(self, cfg_ctrl: BallPoseRos2CtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        # Imported here rather than at module scope so rclpy stays a SOFT dependency: the controller
        # registry only imports this module when a config actually selects BallPoseRos2Ctrl, so
        # Redis-only and sim-only users never need ROS2 installed at all.
        import rclpy
        from geometry_msgs.msg import PointStamped, Vector3Stamped
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._rclpy = rclpy
        self.stale_after_s = cfg_ctrl.stale_after_s

        self._lock = threading.Lock()
        self._ball: np.ndarray | None = None
        self._ball_t: float | None = None  # seconds, in the node clock's domain
        self._aim = np.zeros(2, dtype=np.float32)
        self._aim_received = False

        # rclpy.init() is process-global and raises if called twice -- another component (or an
        # embedding application) may already own the context, so only initialise if nobody has.
        self._owns_rclpy_context = False
        if not rclpy.ok():
            init_kwargs = {}
            if cfg_ctrl.domain_id is not None:
                init_kwargs["domain_id"] = cfg_ctrl.domain_id
            rclpy.init(args=None, **init_kwargs)
            self._owns_rclpy_context = True

        self.node = rclpy.create_node(cfg_ctrl.node_name)
        self._clock = self.node.get_clock()

        qos = QoSProfile(
            depth=cfg_ctrl.qos_depth,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE if cfg_ctrl.qos_reliable else ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.node.create_subscription(PointStamped, cfg_ctrl.ball_topic, self._on_ball, qos)
        self.node.create_subscription(Vector3Stamped, cfg_ctrl.aim_topic, self._on_aim, qos)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._stop = False
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # Block for a first reading, same as BallPoseRedisCtrl. dummy_ball_perception.py publishes
        # immediately using an assumed robot-at-origin pose precisely so this cannot deadlock
        # against its own wait for our robot pose, in either start order.
        waited = 0.0
        while True:
            with self._lock:
                if self._ball is not None:
                    break
            time.sleep(0.01)
            waited += 0.01
            if int(waited) % 2 == 0 and abs(waited - round(waited)) < 0.005:
                logger.warning(
                    f"[BallPoseRos2Ctrl] Still waiting for a first message on '{cfg_ctrl.ball_topic}' "
                    f"({waited:.0f}s) -- is the perception process running? e.g. "
                    f"python scripts/dummy_ball_perception.py --transport ros2. Also check that "
                    f"ROS_DOMAIN_ID and RMW_IMPLEMENTATION match on both sides -- a mismatch looks "
                    f"exactly like 'nobody is publishing'."
                )
        if not self._aim_received:
            logger.warning(
                f"[BallPoseRos2Ctrl] Ball readings are flowing but nothing has been published on "
                f"'{cfg_ctrl.aim_topic}' yet -- kick_target_pos_b stays [0, 0] until something is. "
                f"For a kick_aim_enabled checkpoint at theta=0 that happens to BE the correct value, "
                f"so a mis-wired aim topic is easy to miss here: check 'ros2 topic hz {cfg_ctrl.aim_topic}'."
            )
        logger.info(f"[BallPoseRos2Ctrl] Initialized with first data on '{cfg_ctrl.ball_topic}'.")

    def _spin(self):
        while not self._stop and self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _now(self) -> float:
        return self._clock.now().nanoseconds * 1e-9

    def _on_ball(self, msg):
        # An unset header stamp falls back to arrival time: staleness then measures "how long since
        # WE heard anything", still the property that matters, just without the publisher's send
        # time. Using the node clock (not time.time()) keeps this correct under use_sim_time.
        t = _stamp_to_sec(msg.header.stamp)
        with self._lock:
            self._ball = np.array([msg.point.x, msg.point.y, msg.point.z], dtype=np.float32)
            self._ball_t = t if t is not None else self._now()

    def _on_aim(self, msg):
        with self._lock:
            self._aim = np.array([msg.vector.x, msg.vector.y], dtype=np.float32)
            self._aim_received = True

    def reset(self):
        with self._lock:
            self._ball = None
            self._ball_t = None

    def get_data(self):
        """Latest ball/aim readings; valid=False if the ball reading is missing or older than
        stale_after_s, so the policy falls back to zeros instead of acting on a frozen detection."""
        with self._lock:
            ball = self._ball
            ball_t = self._ball_t
            aim = self._aim
        valid = ball is not None and ball_t is not None and (self._now() - ball_t) <= self.stale_after_s
        return {
            "kick_ball_pos_b": ball,
            "kick_target_pos_b": aim if ball is not None else None,
            "valid": valid,
        }

    def shutdown(self):
        self._stop = True
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self._executor.shutdown()
        self.node.destroy_node()
        # Only tear down the process-global context if we were the ones who created it.
        if self._owns_rclpy_context and self._rclpy.ok():
            self._rclpy.shutdown()
