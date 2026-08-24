"""Standalone dummy "perception" process for RoboJuDo's --live-ball flag (see
run_pipeline_prepared.py and robojudo/controller/ball_pose_redis_ctrl.py /
robojudo/controller/ball_pose_ros2_ctrl.py).

Run this in a SECOND terminal, alongside:
    python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick --live-ball
(add --ball-source ros2 on both sides to use ROS2 instead of Redis for the ball/aim channel --
see TRANSPORT below. The robot/true-ball sim-state relay this script reads from is ALWAYS Redis,
independent of --transport -- see --robot-redis-key.)

WHAT THIS SIMULATES, AND WHAT IT DOESN'T:
By default this TRACKS THE REAL SIMULATED BALL: run_pipeline_prepared.py publishes the physical
ball's true current world position every tick (it moves when the robot's legs/feet push it -- a
fixed/fabricated position would silently drift arbitrarily far from whatever's actually visible in
the viewer the moment any contact happens, which is exactly what made "walk up to the ball you can
see" testing meaningless before this). This script combines that with the robot's own published pose
to compute a correctly-updating robot-frame reading.

This is still a legitimate stand-in for a real detector, not a "cheat": a real onboard perception
stack has no ground-truth access either (there's no such thing as "ground truth" on real hardware),
but it doesn't need any -- its camera is rigidly mounted on the robot, so a raw detection already
reflects wherever the ball visually is RIGHT NOW, in real time, automatically. Reading the true sim
ball position is this script's way of reproducing that same "always reports what's actually there"
property without running real computer vision -- not a shortcut real perception lacks, just a
different (sim-only) way of getting it. --jitter-std and --interactive below still let you layer
REAL perception imperfection (noise, manual mis-detection) on top of that live ground truth, which is
the useful, realistic way to stress-test the policy, instead of a fully disconnected fabrication.

If the ball's true position isn't available (e.g. testing this script against a real-hardware
run_pipeline_prepared.py, which never has a true ball to publish), this falls back to a fixed
fabricated position (--ball-x/-y/-z) -- useful for rehearsing the interface, not for interactive
approach/kick testing.

MODES
  default: tracks the ball's true simulated world position live.
  --jitter-std M   : add per-tick Gaussian noise (std M meters, xy only) to the final robot-frame
                     reading -- simulates detector noise on top of the true position; 0 (default) =
                     perfectly clean.
  --interactive     : also start a small keyboard listener (arrow keys) so you can manually nudge a
                     WORLD-frame offset on top of wherever the ball actually is -- useful for
                     testing what happens when perception is off by a self-chosen amount.
                       Up/Down    : offset away / closer (world x)
                       Left/Right : offset laterally (world y)
                       r          : clear the offset
                     Needs a DISPLAY (pynput's X11 backend), same constraint as RoboJuDo's own
                     KeyboardCtrl.

KICK_TARGET_POS_B: WORLD-FRAME OFFSET vs. AZIMUTH COMMAND (read this before deploying a
2026-08-22-or-later checkpoint)
  This script has always published `kick_target_pos_b` as a live world-frame transform (rel_target
  rotated into the robot's heading frame) -- correct for a checkpoint trained BEFORE the
  azimuth-aim refactor, or any checkpoint whose skill has SkillConfig.kick_aim_enabled=False.
  UnifiedLocoKickPolicy (RoboJuDo) forwards whatever this script publishes straight into that same
  261-dim observation slot regardless of which semantics the loaded ONNX actually expects -- there
  is currently no metadata in the ONNX export that would let it tell the difference, so getting
  this flag right is entirely on the caller. See mujoco_kick_rollout_worker.py's own
  --kick-aim-enabled flag (same project, same refactor, the reference implementation this mirrors)
  for the full azimuth-aim background.

  For a checkpoint trained WITH kick_aim_enabled=True (every skill in playground/
  unified_ball_kick_enhanced as of 2026-08-24), the policy instead expects a CONSTANT, bounded
  command -- [kick_aim_theta_deg / kick_aim_theta_ref_deg, 0.0] -- held fixed for the whole kick
  attempt, not a live per-tick metre-scale offset. Feeding the old world-frame transform to such a
  checkpoint is silently wrong, not a crash: the observation vector's shape and slot position are
  unchanged (the refactor deliberately kept the term named "kick_target_pos_b" so old checkpoints
  keep warm-starting), but the VALUE is typically 15-20x the trained magnitude and varies every
  tick instead of staying constant.

  --kick-aim-enabled           : this checkpoint was trained with SkillConfig.kick_aim_enabled=True
                                  -- publish the bounded aim command instead of the world-frame
                                  target_pos_b transform. Default off (unchanged legacy behavior).
  --kick-aim-theta-deg D       : the fixed kick_aim_theta (degrees) to publish when
                                  --kick-aim-enabled is set. 0.0 (default) aims straight along the
                                  skill's own calibrated nominal bearing. Ignored otherwise.
  --kick-aim-theta-ref-deg D   : normalization reference (degrees), must match the checkpoint's own
                                  MultiSkillConfig/BallConfig.kick_aim_theta_ref_deg -- 45.0 is that
                                  field's own default and, per its docstring, is meant to stay fixed
                                  across curriculum changes. Ignored when --kick-aim-enabled is not
                                  set.
  --target-x/--target-y are ignored entirely in this mode -- there is no world-frame target point
  to compute a relative offset from (kick_aim_theta is a bearing OFFSET, not a point).

TRANSPORT: --transport {redis, ros2}
  'redis' (default, unchanged): SET the ball/aim reading as one JSON blob to --redis-key, matching
  BallPoseRedisCtrl.

  'ros2': publish kick_ball_pos_b as geometry_msgs/PointStamped on --ball-topic, and
  kick_target_pos_b as geometry_msgs/Vector3Stamped (.x/.y used) on --aim-topic, matching
  BallPoseRos2Ctrl -- see that module's docstring for why these are two separate topics (a
  perception signal that must keep flowing vs. a held command that deliberately does not expire)
  and its QoS default (BEST_EFFORT/VOLATILE, depth 5). Requires rclpy (ROS2) to be importable;
  imported lazily so plain --transport redis runs never need it installed.

  Either transport, --robot-redis-key is STILL READ OVER REDIS: it is a sim-only convenience (see
  this module's own WHAT THIS SIMULATES section) with no real-hardware equivalent, so it never
  needs to match whatever --transport the ball/aim channel itself uses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import redis
from redis.exceptions import RedisError

from robojudo.utils.util_func import calc_heading_quat_np, quat_rotate_inverse_np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dummy_ball_perception")

# FALLBACK-ONLY world position, used solely when the publisher has no true ball to report (e.g. a
# real-hardware run_pipeline_prepared.py). Matches training's ball-spawn convention (see
# playground/locomotion_and_ball_kicking/configs/ball_stageC_warmup.yaml: x=2.84, y=-0.46) as a
# sane default. Ignored entirely whenever a true ball_pos_w is being published.
DEFAULT_BALL_XYZ = (2.84, -0.46, 0.11)
DEFAULT_TARGET_XY = (7.84, -0.46)

# FALLBACK-ONLY robot pose, used solely before the first real sim-state reading arrives on
# --robot-redis-key. Without this, this script would block waiting for a robot pose that
# run_pipeline_prepared.py never publishes until ITS BallPoseRedisCtrl has already read a first
# ball-pose reading from US -- a mutual wait that deadlocks both processes regardless of start
# order. Matches the sim's own default base pose (world origin, identity heading).
DEFAULT_BASE_POS_W = (0.0, 0.0, 0.0)
DEFAULT_BASE_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

_ARROW_STEP_M = 0.02  # meters moved per tick while an arrow key is held, in --interactive mode


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--transport",
        choices=["redis", "ros2"],
        default="redis",
        help="How to publish kick_ball_pos_b/kick_target_pos_b -- must match "
        "run_pipeline_prepared.py's --ball-source on the other side. See this module's TRANSPORT "
        "docstring section. The robot/true-ball sim-state relay below is unaffected -- always Redis.",
    )
    ap.add_argument("--redis-host", default="localhost", help="Used for BOTH transports: also serves the "
        "sim-only robot/true-ball relay (--robot-redis-key) regardless of --transport.")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--redis-db", type=int, default=0)
    ap.add_argument(
        "--redis-key", default="ball_pose_g1",
        help="Only used with --transport redis. Must match BallPoseRedisCtrlCfg.redis_key.",
    )
    ap.add_argument(
        "--robot-redis-key",
        default="robot_pose_g1",
        help="Used with EITHER --transport (this relay is always Redis -- see the TRANSPORT "
        "docstring section). Must match run_pipeline_prepared.py's --robot-redis-key. Robot (and "
        "true ball) pose read from here.",
    )
    ap.add_argument(
        "--ball-topic", default="/ball_pose",
        help="Only used with --transport ros2. Must match BallPoseRos2CtrlCfg.ball_topic.",
    )
    ap.add_argument(
        "--aim-topic", default="/kick_aim",
        help="Only used with --transport ros2. Must match BallPoseRos2CtrlCfg.aim_topic.",
    )
    ap.add_argument(
        "--ros2-node-name", default="dummy_ball_perception",
        help="Only used with --transport ros2. rclpy node name for this process's publisher.",
    )
    ap.add_argument(
        "--ros2-domain-id", type=int, default=None,
        help="Only used with --transport ros2. Overrides ROS_DOMAIN_ID for this process; default "
        "(unset) inherits the environment variable, same as any other ROS2 node.",
    )
    ap.add_argument("--rate", type=float, default=30.0, help="Publish rate, Hz.")
    ap.add_argument("--ball-x", type=float, default=DEFAULT_BALL_XYZ[0], help="Fallback ball world x (m) if no true ball.")
    ap.add_argument("--ball-y", type=float, default=DEFAULT_BALL_XYZ[1], help="Fallback ball world y (m) if no true ball.")
    ap.add_argument("--ball-z", type=float, default=DEFAULT_BALL_XYZ[2], help="Fallback ball world z (m) if no true ball.")
    ap.add_argument("--target-x", type=float, default=DEFAULT_TARGET_XY[0], help="Fabricated target world x (m).")
    ap.add_argument("--target-y", type=float, default=DEFAULT_TARGET_XY[1], help="Fabricated target world y (m).")
    ap.add_argument("--jitter-std", type=float, default=0.0, help="Per-tick Gaussian noise std (m, xy only).")
    ap.add_argument("--interactive", action="store_true", help="Arrow-key world-frame offset on top of the ball.")
    ap.add_argument(
        "--kick-aim-enabled", action="store_true",
        help="2026-08-24: this checkpoint was trained with SkillConfig.kick_aim_enabled=True -- "
        "publish kick_target_pos_b as the bounded [kick_aim_theta/kick_aim_theta_ref_deg, 0.0] "
        "command it actually trained on, instead of the pre-azimuth-refactor world-frame "
        "target_pos_b transform (a ~15-20x out-of-distribution magnitude for such a checkpoint -- "
        "see this module's own docstring). Default off, so a checkpoint trained WITHOUT "
        "kick_aim_enabled (or before this refactor) is completely unaffected. --target-x/--target-y "
        "are ignored entirely when this is set.",
    )
    ap.add_argument(
        "--kick-aim-theta-deg", type=float, default=0.0,
        help="The fixed kick_aim_theta value (degrees) to publish when --kick-aim-enabled is set -- "
        "0.0 (default) aims straight along the skill's own calibrated nominal bearing, the natural "
        "choice for a repeatable test (no reason to prefer a random angle here). Ignored entirely "
        "when --kick-aim-enabled is not set.",
    )
    ap.add_argument(
        "--kick-aim-theta-ref-deg", type=float, default=45.0,
        help="Normalization reference (degrees) matching the checkpoint's own MultiSkillConfig/"
        "BallConfig.kick_aim_theta_ref_deg -- 45.0 is that field's own default and, per its "
        "docstring, is meant to stay fixed across curriculum changes, so this default is correct "
        "for every checkpoint in this project unless a config explicitly overrode it (none "
        "currently do). Ignored when --kick-aim-enabled is not set.",
    )
    return ap.parse_args()


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


class _ArrowKeyOffset:
    """Tracks a running (dx, dy) world-frame offset, nudged by held arrow keys. Isolated from
    RoboJuDo's own KeyboardCtrl (which drives the ROBOT's velocity command in the OTHER
    terminal/process) -- this is a separate listener in this process, applied on top of wherever the
    ball actually is (true position, or the fallback if there's no true one)."""

    def __init__(self):
        self.dx = 0.0
        self.dy = 0.0
        self._held = {"up": False, "down": False, "left": False, "right": False}

    def start(self):
        from pynput import keyboard

        def _name(key):
            k = str(key).replace("Key.", "")
            return k if k in self._held else None

        def on_press(key):
            name = _name(key)
            if name is not None:
                self._held[name] = True
            elif getattr(key, "char", None) == "r":
                self.dx = 0.0
                self.dy = 0.0
                logger.info("[interactive] offset cleared")

        def on_release(key):
            name = _name(key)
            if name is not None:
                self._held[name] = False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        logger.info("[interactive] arrow keys offset the ball reading; 'r' clears it. Needs DISPLAY.")

    def step(self):
        self.dx += _ARROW_STEP_M * (float(self._held["up"]) - float(self._held["down"]))
        self.dy += _ARROW_STEP_M * (float(self._held["left"]) - float(self._held["right"]))


def _fetch_sim_state(client: redis.Redis, key: str) -> dict | None:
    raw = client.get(key)
    if raw is None:
        return None
    payload = json.loads(raw)
    ball_pos_w = payload.get("ball_pos_w")
    return {
        "base_pos_w": np.asarray(payload["base_pos_w"], dtype=np.float64),
        "base_quat_xyzw": np.asarray(payload["base_quat_xyzw"], dtype=np.float64),
        "ball_pos_w": np.asarray(ball_pos_w, dtype=np.float64) if ball_pos_w is not None else None,
        "t": float(payload["t"]),
    }


class _Ros2BallPublisher:
    """--transport ros2 counterpart of the plain `client.set(args.redis_key, payload)` call in the
    --transport redis path -- publishes the same two numbers (ball_pos_b, target_pos_b) as two
    topics instead of one JSON blob. See BallPoseRos2Ctrl's module docstring for why it's two
    topics and the QoS rationale (mirrored here so publisher and subscriber always agree)."""

    def __init__(self, node_name: str, ball_topic: str, aim_topic: str, domain_id: int | None):
        # Imported here, not at module scope, so `--transport redis` (the default) never needs
        # ROS2 installed at all -- matches BallPoseRos2Ctrl's own lazy-import rationale.
        import rclpy
        from geometry_msgs.msg import PointStamped, Vector3Stamped
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        self._rclpy = rclpy
        self._PointStamped = PointStamped
        self._Vector3Stamped = Vector3Stamped

        self._owns_rclpy_context = False
        if not rclpy.ok():
            init_kwargs = {} if domain_id is None else {"domain_id": domain_id}
            rclpy.init(args=None, **init_kwargs)
            self._owns_rclpy_context = True

        self.node = rclpy.create_node(node_name)
        # BEST_EFFORT/VOLATILE: matches BallPoseRos2Ctrl's default subscriber QoS exactly. A
        # RELIABLE publisher would still connect fine to a BEST_EFFORT subscriber (QoS compatibility
        # is about the subscriber not being STRICTER than the publisher), but there's no reason for
        # this dummy publisher to ask for delivery guarantees the consumer doesn't need.
        qos = QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._ball_pub = self.node.create_publisher(PointStamped, ball_topic, qos)
        self._aim_pub = self.node.create_publisher(Vector3Stamped, aim_topic, qos)

    def publish(self, ball_pos_b: np.ndarray, target_pos_b: np.ndarray) -> None:
        stamp = self.node.get_clock().now().to_msg()

        ball_msg = self._PointStamped()
        ball_msg.header.stamp = stamp
        ball_msg.header.frame_id = "base_link"
        ball_msg.point.x, ball_msg.point.y, ball_msg.point.z = (float(v) for v in ball_pos_b)
        self._ball_pub.publish(ball_msg)

        aim_msg = self._Vector3Stamped()
        aim_msg.header.stamp = stamp
        aim_msg.header.frame_id = "base_link"
        aim_msg.vector.x, aim_msg.vector.y = float(target_pos_b[0]), float(target_pos_b[1])
        aim_msg.vector.z = 0.0
        self._aim_pub.publish(aim_msg)

    def shutdown(self) -> None:
        self.node.destroy_node()
        if self._owns_rclpy_context and self._rclpy.ok():
            self._rclpy.shutdown()


def main() -> int:
    args = parse_args()

    if args.interactive and not os.environ.get("DISPLAY"):
        raise RuntimeError(
            "--interactive needs a DISPLAY (pynput's keyboard backend requires an X server), same "
            "constraint as RoboJuDo's own KeyboardCtrl. Drop --interactive for a headless run."
        )

    # ALWAYS connect to Redis, regardless of --transport: this client also drives the sim-only
    # robot/true-ball relay (--robot-redis-key), which has no ROS2 counterpart -- see this module's
    # TRANSPORT docstring section.
    client = connect_redis(args.redis_host, args.redis_port, args.redis_db)

    ros2_pub = None
    if args.transport == "ros2":
        ros2_pub = _Ros2BallPublisher(
            node_name=args.ros2_node_name,
            ball_topic=args.ball_topic,
            aim_topic=args.aim_topic,
            domain_id=args.ros2_domain_id,
        )
        logger.info(
            f"[transport=ros2] Publishing ball_pos_b -> '{args.ball_topic}' (PointStamped), "
            f"target_pos_b -> '{args.aim_topic}' (Vector3Stamped) at {args.rate:.0f} Hz."
        )

    arrows = _ArrowKeyOffset()
    if args.interactive:
        arrows.start()

    period = 1.0 / args.rate
    rng = np.random.default_rng()
    tick = 0
    aim_mode = args.kick_aim_enabled
    if aim_mode:
        aim_command = np.array(
            [args.kick_aim_theta_deg / args.kick_aim_theta_ref_deg, 0.0], dtype=np.float32
        )
        logger.info(
            f"Publishing to '{args.redis_key}' at {args.rate:.0f} Hz -- tracking the true simulated ball "
            f"when available, else falling back to a fixed world pos=({args.ball_x:.2f}, {args.ball_y:.2f}, "
            f"{args.ball_z:.2f}). --kick-aim-enabled: publishing the CONSTANT aim command "
            f"kick_target_pos_b={aim_command.tolist()} (kick_aim_theta_deg={args.kick_aim_theta_deg:.1f}, "
            f"kick_aim_theta_ref_deg={args.kick_aim_theta_ref_deg:.1f}) instead of a world-frame target "
            f"transform -- --target-x/--target-y are ignored. Reading sim state from "
            f"'{args.robot_redis_key}' -- assumed robot-at-origin until that's available, to avoid "
            f"deadlocking with run_pipeline_prepared.py's own wait for a first reading from us."
        )
    else:
        logger.info(
            f"Publishing to '{args.redis_key}' at {args.rate:.0f} Hz -- tracking the true simulated ball "
            f"when available, else falling back to a fixed world pos=({args.ball_x:.2f}, {args.ball_y:.2f}, "
            f"{args.ball_z:.2f}). Target world pos=({args.target_x:.2f}, {args.target_y:.2f}). Reading sim "
            f"state from '{args.robot_redis_key}' -- assumed robot-at-origin until that's available, to "
            f"avoid deadlocking with run_pipeline_prepared.py's own wait for a first reading from us."
        )

    _warned_no_true_ball = False
    _have_real_sim_state = False
    _no_sim_state_waited = 0.0

    try:
        while True:
            t_start = time.time()
            arrows.step()

            sim_state = _fetch_sim_state(client, args.robot_redis_key)
            if sim_state is not None:
                if not _have_real_sim_state:
                    logger.info(
                        f"Got first real sim state from '{args.robot_redis_key}' -- switching to true "
                        "robot-frame readings."
                    )
                    _have_real_sim_state = True
            elif _have_real_sim_state:
                # Publisher went away (process stopped/crashed) after previously being up -- rather
                # than freeze on the last-known pose (which would silently reintroduce the exact
                # "stale but reported as live" bug this whole redesign fixes), skip the tick:
                # BallPoseRedisCtrl will see OUR "t" stop refreshing and correctly fall back to zero.
                time.sleep(period)
                continue
            else:
                # No real sim state has arrived yet -- e.g. run_pipeline_prepared.py isn't up, or
                # (the common case) it IS up but its BallPoseRedisCtrl is itself blocked waiting on
                # OUR first ball-pose reading, which would otherwise deadlock the two processes
                # forever regardless of start order. Publish now using an assumed
                # robot-at-origin pose so a first reading is available immediately; this switches to
                # the real robot pose the moment run_pipeline_prepared.py starts publishing it.
                _no_sim_state_waited += period
                if abs(_no_sim_state_waited % 2.0) < period:
                    logger.warning(
                        f"No sim state yet on '{args.robot_redis_key}' ({_no_sim_state_waited:.0f}s) -- "
                        "publishing with an assumed robot-at-origin pose until run_pipeline_prepared.py "
                        "is up (is it running with --live-ball?)."
                    )
                sim_state = {
                    "base_pos_w": np.array(DEFAULT_BASE_POS_W, dtype=np.float64),
                    "base_quat_xyzw": np.array(DEFAULT_BASE_QUAT_XYZW, dtype=np.float64),
                    "ball_pos_w": None,
                    "t": t_start,
                }

            if sim_state["ball_pos_w"] is not None:
                ball_pos_w = sim_state["ball_pos_w"] + np.array([arrows.dx, arrows.dy, 0.0])
            else:
                if not _warned_no_true_ball:
                    logger.warning(
                        "No true ball_pos_w in the sim-state payload -- falling back to the fixed "
                        f"({args.ball_x}, {args.ball_y}, {args.ball_z}) position (expected on real "
                        "hardware; unexpected in sim -- check --live-ball's ball_freejoint lookup)."
                    )
                    _warned_no_true_ball = True
                ball_pos_w = np.array([args.ball_x + arrows.dx, args.ball_y + arrows.dy, args.ball_z])
            heading_quat = calc_heading_quat_np(sim_state["base_quat_xyzw"])
            ball_pos_b = quat_rotate_inverse_np(heading_quat, ball_pos_w - sim_state["base_pos_w"])

            if aim_mode:
                # Constant, world-frame-independent command -- kick_aim_theta is a bearing OFFSET
                # sampled once per attempt and held, not a live measurement of anything, so it does
                # NOT depend on sim_state/heading/ball position at all (mirrors
                # managers/observation/terms/unified.py::kick_aim_command's kick_aim_enabled=True
                # branch exactly). target_x/y are unused in this mode.
                target_pos_b = aim_command
            else:
                target_pos_w = np.array([args.target_x, args.target_y, ball_pos_w[2]])
                target_pos_b = quat_rotate_inverse_np(heading_quat, target_pos_w - sim_state["base_pos_w"])[:2]

            if args.jitter_std > 0:
                ball_pos_b = ball_pos_b.copy()
                ball_pos_b[:2] += rng.normal(0.0, args.jitter_std, size=2)

            if args.transport == "ros2":
                ros2_pub.publish(ball_pos_b, target_pos_b)
            else:
                payload = json.dumps(
                    {
                        "kick_ball_pos_b": ball_pos_b.tolist(),
                        "kick_target_pos_b": target_pos_b.tolist(),
                        "t": t_start,
                    }
                )
                try:
                    client.set(args.redis_key, payload)
                except RedisError as e:
                    logger.warning(f"Lost redis connection ({e}), reconnecting...")
                    client = connect_redis(args.redis_host, args.redis_port, args.redis_db)

            tick += 1
            if tick % (int(args.rate) * 2) == 0:  # log roughly every 2s
                logger.info(f"ball_pos_b={np.round(ball_pos_b, 3).tolist()} target_pos_b={np.round(target_pos_b, 3).tolist()}")

            elapsed = time.time() - t_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    finally:
        if ros2_pub is not None:
            ros2_pub.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
