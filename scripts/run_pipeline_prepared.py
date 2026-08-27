# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
import logging
import time

import numpy as np
import redis
from redis.exceptions import RedisError

# import robojudo.pipeline (and therefore torch, via robojudo/__init__.py) BEFORE mujoco. On some
# platforms (seen on the G1 onboard computer) importing mujoco first exhausts glibc's static TLS
# surplus -- a small, fixed-size-per-process area that native-extension shared libraries claim on
# load -- leaving no room for torch's own bundled libgomp.so.1 to claim its block, which fails with
# "cannot allocate memory in static TLS block". The stock run_pipeline.py never imports mujoco at
# module scope at all, so it never hit this; keep torch loading first here too.
import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.config.g1.env.g1_holosoma_env_cfg import HOLOSOMA_SCENE_WITH_BALL
from robojudo.controller.ctrl_cfgs import BallPoseRedisCtrlCfg, BallPoseRos2CtrlCfg, BallPoseUdpCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

import mujoco  # noqa: E402 -- see note above; must come after robojudo.pipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the full policy loop (observations, actions, logging) every tick, but never "
            "apply torque to the robot/sim -- nothing physically moves (env.step() is skipped, "
            "same mechanism self_check() already uses internally for its 10-step warmup). Use "
            "this to review a checkpoint's commanded actions -- with keyboard/joystick commands "
            "still live -- before letting it actually move, especially on real hardware. Unsafe- "
            "looking actions (NaN, action-clip saturation, a commanded joint angle outside the "
            "model's range, a would-be torque beyond the joint's torque limit, or a large tick-to-"
            "tick action jump) are logged as warnings."
        ),
    )
    parser.add_argument(
        "--live-ball",
        action="store_true",
        help=(
            "Feed the policy LIVE kick_ball_pos_b/kick_target_pos_b (instead of the hardcoded "
            "zero) by adding a live-ball controller to the pipeline's controllers -- see "
            "--ball-source for which transport (Redis or ROS2) it uses to reach a separate "
            "perception process running in another terminal: scripts/dummy_ball_perception.py for "
            "sim testing today, a real onboard detector publishing to the same channel/shape on the "
            "robot later. In sim, also swaps in scene_g1_29dof_with_ball.xml (a physical ball to "
            "actually kick) and PUBLISHES this process's own robot pose, plus the ball's TRUE "
            "current world position (it moves when the robot's legs/feet push it -- reporting a "
            "fixed spawn point would silently drift arbitrarily far from what's visible in the "
            "viewer), over Redis to --robot-redis-key every tick -- this sim-only relay always uses "
            "Redis regardless of --ball-source, since it exists purely to let dummy_ball_perception.py "
            "compute a robot-frame reading in sim and has no real-hardware counterpart at all (a real "
            "camera-based detector reports relative to itself for free, with no ground truth to relay "
            "either). On real hardware there's no MuJoCo scene to swap and nothing true to publish, "
            "so --live-ball there only wires the controller chosen by --ball-source."
        ),
    )
    parser.add_argument(
        "--ball-source",
        type=str,
        choices=["redis", "ros2", "udp"],
        default="redis",
        help=(
            "Only used with --live-ball. Transport for kick_ball_pos_b/kick_target_pos_b: "
            "'redis' (default, unchanged) adds a BallPoseRedisCtrl -- see --redis-host/--ball-redis-key. "
            "'ros2' adds a BallPoseRos2Ctrl subscribing to two topics instead (a live ball-position "
            "topic plus a held aim-command topic) -- see --ball-topic/--aim-topic/--ros2-*. "
            "'udp' adds a BallPoseUdpCtrl listening on a plain UDP socket -- stdlib-only on both "
            "ends, no Redis and no direct cross-distro ROS2 needed; use this for real-hardware "
            "deployment where Redis cannot be installed/run at all (confirmed on a real G1) -- see "
            "--udp-listen-host/--udp-listen-port and scripts/foxy_ros2_ball_bridge.py, which relays "
            "real ROS2 topics into this same UDP channel from inside the robot's native Foxy. Point "
            "scripts/dummy_ball_perception.py at the same transport with its own --transport flag "
            "(redis/ros2 sim testing only -- udp has no dummy-perception counterpart, since it's "
            "meant to be fed by foxy_ros2_ball_bridge.py or a real detector, not simulated)."
        ),
    )
    parser.add_argument(
        "--redis-host",
        type=str,
        default=BallPoseRedisCtrlCfg().redis_host,
        help=(
            "Used with --live-ball (always, for the sim-only robot-pose relay) and with "
            "--ball-source redis (for the ball-pose channel itself)."
        ),
    )
    parser.add_argument(
        "--ball-redis-key",
        type=str,
        default=BallPoseRedisCtrlCfg().redis_key,
        help="Only used with --live-ball --ball-source redis. Redis key the perception process publishes ball pose to.",
    )
    parser.add_argument(
        "--robot-redis-key",
        type=str,
        default="robot_pose_g1",
        help=(
            "Only used with --live-ball in sim (always over Redis, independent of --ball-source -- "
            "see --live-ball's help). Redis key this process publishes its own base_pos_w/"
            "base_quat_xyzw to every tick, for the perception process to consume."
        ),
    )
    parser.add_argument(
        "--ball-topic",
        type=str,
        default=BallPoseRos2CtrlCfg().ball_topic,
        help="Only used with --live-ball --ball-source ros2. geometry_msgs/PointStamped topic the "
        "perception process publishes the live ball position to (robot heading frame, x-fwd/z-up).",
    )
    parser.add_argument(
        "--aim-topic",
        type=str,
        default=BallPoseRos2CtrlCfg().aim_topic,
        help="Only used with --live-ball --ball-source ros2. geometry_msgs/Vector3Stamped topic "
        "carrying the held kick_target_pos_b/aim command (.x/.y used) -- see BallPoseRos2Ctrl's "
        "module docstring for why this is a separate, non-expiring topic from --ball-topic.",
    )
    parser.add_argument(
        "--ros2-node-name",
        type=str,
        default=BallPoseRos2CtrlCfg().node_name,
        help="Only used with --live-ball --ball-source ros2. rclpy node name for this process's subscriber.",
    )
    parser.add_argument(
        "--ros2-domain-id",
        type=int,
        default=None,
        help="Only used with --live-ball --ball-source ros2. Overrides ROS_DOMAIN_ID for this "
        "process; default (unset) inherits the environment variable, same as any other ROS2 node.",
    )
    parser.add_argument(
        "--udp-listen-host",
        type=str,
        default=BallPoseUdpCtrlCfg().listen_host,
        help="Only used with --live-ball --ball-source udp. Interface this process listens on "
        "for ball/aim datagrams (0.0.0.0 = all interfaces).",
    )
    parser.add_argument(
        "--udp-listen-port",
        type=int,
        default=BallPoseUdpCtrlCfg().listen_port,
        help="Only used with --live-ball --ball-source udp. Must match the sender's "
        "--udp-dest-port (foxy_ros2_ball_bridge.py or dummy_ball_perception.py --transport udp).",
    )
    parser.add_argument(
        "--ball-log-hz",
        type=float,
        default=1.0,
        help="Only used with --live-ball. Rate (Hz) to log kick_ball_pos_b/kick_target_pos_b as "
        "actually RECEIVED by this process's live-ball controller (not what the sender intended to "
        "publish -- reflects real staleness/validity too). 0 disables this logging.",
    )
    parser.add_argument(
        "--ball-log-decimals",
        type=int,
        default=5,
        help="Only used with --ball-log-hz. Decimal places to print ball_pos_b/target_pos_b to. "
        "Formatted as a fixed-precision string, not np.round()+tolist() -- the latter doesn't "
        "actually shorten float32 values (e.g. -0.682 stored as float32 widens to Python's float64 "
        "as -0.6819999814033508 regardless of rounding first; only string formatting truncates the "
        "DISPLAY cleanly).",
    )
    parser.add_argument(
        "--kick-aim-theta-ref-deg",
        type=float,
        default=45.0,
        help="Only used with --live-ball, for the --ball-log-hz logging's DISPLAY-ONLY theta_if_aim "
        "conversion (kick_target_pos_b[0] * this) -- does not affect the policy at all. Must match "
        "the sender's own --kick-aim-theta-ref-deg (dummy_ball_perception.py) if you want the shown "
        "degrees to be meaningful; this process has no way to know if the checkpoint actually uses "
        "kick_aim_enabled, so the shown value is only correct when that's true. Default 45.0 matches "
        "this project's own MultiSkillConfig/BallConfig.kick_aim_theta_ref_deg default.",
    )
    args = parser.parse_args()
    return args


class BallRespawner:
    """--live-ball only: teleports the physical MuJoCo ball back to a nominal spawn position every
    time a kick ends (auto-return-after-clip-finishes, or a manual [RETURN_TO_LOCO]), so the NEXT
    kick -- whether it repeats the same skill or a different one via [CYCLE_KICK_SKILL]/`j` -- always
    starts from an in-distribution ball position, matching training's per-episode ball reset. Without
    this, the ball just stays wherever physics left it after being kicked (often far downfield, near
    the target) -- badly out-of-distribution for the next kick, and the actual cause of "robust the
    1st time, falls the 2nd" reports (confirmed: [CYCLE_KICK_SKILL] itself never touches env state --
    see unified_loco_kick_policy.py's docstring -- so this was never a cycling bug).

    Detects the kick-end transition by polling `inner_policy.task_mode` tick-to-tick (same pattern
    DryRunSafetyMonitor already uses for the same field) rather than adding an event/callback to
    UnifiedLocoKickPolicy itself, which stays environment-agnostic (it never touches env.data/model,
    so it also works unmodified against real hardware, where there's obviously nothing to teleport).
    """

    DEFAULT_BALL_XY = (1.3, 0.0)  # scene_g1_29dof_with_ball.xml's own ball body spawn (x, y)
    BALL_Z = 0.11  # ball radius -- resting exactly on the floor, same as the scene's spawn/keyframe

    def __init__(self, pipeline, ball_qpos_addr: int, ball_qvel_addr: int):
        self.env = pipeline.env
        self.inner_policy = pipeline.policy.policy  # UnifiedLocoKickPolicy (unwrap PolicyWrapper)
        self.ball_qpos_addr = ball_qpos_addr
        self.ball_qvel_addr = ball_qvel_addr
        self._prev_task_mode = self.inner_policy.task_mode

    def check_and_respawn(self):
        task_mode = self.inner_policy.task_mode
        if self._prev_task_mode == "kick" and task_mode == "locomotion":
            self._respawn()
        self._prev_task_mode = task_mode

    def _respawn(self):
        skill_id = self.inner_policy.kick_skill_id
        xy = self.inner_policy.get_skill_ball_xy(skill_id)
        x, y = xy if xy is not None else self.DEFAULT_BALL_XY

        addr = self.ball_qpos_addr
        self.env.data.qpos[addr : addr + 3] = [x, y, self.BALL_Z]
        self.env.data.qpos[addr + 3 : addr + 7] = [1.0, 0.0, 0.0, 0.0]  # identity quat, wxyz
        vaddr = self.ball_qvel_addr
        self.env.data.qvel[vaddr : vaddr + 6] = 0.0  # no residual roll/spin from the last kick

        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.update()
        logger.info(f"[live-ball] kick ended (skill {skill_id}) -- respawned ball at ({x:.2f}, {y:.2f}, {self.BALL_Z:.2f})")


_BALL_CTRL_TYPE_BY_SOURCE = {"redis": "BallPoseRedisCtrl", "ros2": "BallPoseRos2Ctrl", "udp": "BallPoseUdpCtrl"}


def _fmt_vec(values, decimals: int) -> str:
    """Fixed-precision display of a float32 vector for logging. NOT np.round(values, decimals)
    .tolist() -- that widens each already-imprecise float32 element to a Python float64, which
    prints with the float32 value's full binary noise regardless of the rounding (e.g. a float32
    holding -0.682 widens to -0.6819999814033508). Explicit string formatting truncates the DISPLAY
    cleanly instead, independent of the underlying binary representation."""
    return "[" + ", ".join(f"{float(v):.{decimals}f}" for v in values) + "]"


def _peek_ball_reading(ctrl_inst, ctrl_type: str):
    """Read-only snapshot of a live-ball controller's latest reading, for --ball-log-hz logging.
    Deliberately does NOT call ctrl_inst.get_data() for the Redis/UDP controllers: their get_data()
    pops from an internal deque, and this function is called independently of (and at a different
    rate than) the pipeline's own per-tick get_data() call inside policy.get_observation() -- a
    second, uncoordinated pop from the same deque would race with that real consumption and could
    steal a reading meant for the policy. Instead reads their cached `last_data` snapshot (set by
    whichever call -- ours or the pipeline's -- most recently popped), which is safe: pure
    inspection, no side effects. BallPoseRos2Ctrl is different -- lock-protected fields overwritten
    by ROS2 callbacks, get_data() is already a pure read -- so it's called directly there.

    Returns None if nothing has ever arrived, else (ball_pos_b, target_pos_b, age_seconds_or_None,
    valid). age is None for ros2 (that controller's get_data() doesn't expose the raw timestamp,
    only the pre-computed valid bool)."""
    if ctrl_type == "BallPoseRos2Ctrl":
        data = ctrl_inst.get_data()
        if data["kick_ball_pos_b"] is None:
            return None
        return data["kick_ball_pos_b"], data["kick_target_pos_b"], None, data["valid"]
    last = ctrl_inst.last_data
    if last is None:
        return None
    age = time.time() - last["t"]
    valid = age <= ctrl_inst.stale_after_s
    return last["kick_ball_pos_b"], last["kick_target_pos_b"], age, valid


class DryRunSafetyMonitor:
    """Post-step safety checks for --dry-run. pipeline.step(dry_run=True) skips env.step() (so
    nothing physically moves) but still runs get_observation/get_pd_target every tick -- this
    wraps pipeline.policy.get_pd_target to capture each tick's commanded target (pipeline.step()
    doesn't return it) and checks it against known-unsafe conditions.

    Because dry_run also means the physical state (dof_pos/dof_vel) never advances, the
    torque/range checks reflect "if this action were applied to the CURRENT (frozen) pose" -- they
    won't catch problems that only appear after the robot has actually moved. This is a pre-flight
    sanity check on the policy's commanded actions, not a substitute for a real (careful) run.
    """

    ACTION_CLIP_SATURATION_FRAC = 0.95  # fraction of action_clip counted as "saturating"
    ACTION_RATE_WARN_RAD = 5.0  # raw (pre action_scale) action jump per tick

    def __init__(self, pipeline):
        self.env = pipeline.env
        self.inner_policy = pipeline.policy.policy
        self._tick = 0
        self._last_pd_target = None
        self._prev_raw_action = None

        self._joint_names = list(self.env.dof_cfg.joint_names)
        self._joint_ranges = [None] * len(self._joint_names)
        # env.model (raw MuJoCo model) only exists on MujocoEnv (sim) -- UnitreeCppEnv (real) has
        # no such attribute, since it wraps the real robot's SDK directly. Both env configs load a
        # MuJoCo model for forward-kinematics purposes regardless (same XML), exposed uniformly as
        # env.kinematics.model on the base Environment class -- use that instead so joint-range
        # checks work on both sim and real. Falls back to skipping the range check (still get
        # torque/NaN/saturation/rate checks) if kinematics wasn't configured for some other env.
        kin_model = getattr(getattr(self.env, "kinematics", None), "model", None)
        if kin_model is not None:
            for i, name in enumerate(self._joint_names):
                jid = mujoco.mj_name2id(kin_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                limited = jid >= 0 and bool(kin_model.jnt_limited[jid])
                self._joint_ranges[i] = tuple(kin_model.jnt_range[jid]) if limited else None
        else:
            logger.warning(
                "DryRunSafetyMonitor: env has no forward-kinematics model available -- skipping "
                "joint-range checks (NaN/saturation/torque-limit/action-rate checks still run)."
            )

        # get_pd_target's return value is never surfaced by pipeline.step(), so capture it here.
        orig_get_pd_target = pipeline.policy.get_pd_target

        def _capturing_get_pd_target(obs):
            pd_target = orig_get_pd_target(obs)
            self._last_pd_target = np.asarray(pd_target, dtype=np.float64)
            return pd_target

        pipeline.policy.get_pd_target = _capturing_get_pd_target

    def check_and_log(self):
        self._tick += 1
        pd_target = self._last_pd_target
        if pd_target is None:
            return
        raw_action = np.asarray(self.inner_policy.last_action, dtype=np.float64)  # pre-scale

        issues = []

        if not np.all(np.isfinite(pd_target)) or not np.all(np.isfinite(raw_action)):
            issues.append("NaN/Inf in policy output")

        action_clip = self.inner_policy.action_clip
        if action_clip:
            sat = np.abs(raw_action) >= self.ACTION_CLIP_SATURATION_FRAC * action_clip
            if sat.any():
                names = [self._joint_names[i] for i in np.where(sat)[0]]
                issues.append(f"raw action saturating action_clip (±{action_clip:.0f}) on: {names}")

        dof_pos = np.asarray(self.env.dof_pos, dtype=np.float64)
        dof_vel = np.asarray(self.env.dof_vel, dtype=np.float64)
        stiffness = np.asarray(self.env.stiffness, dtype=np.float64)
        damping = np.asarray(self.env.damping, dtype=np.float64)
        torque_limits = np.asarray(self.env.torque_limits, dtype=np.float64)
        would_be_torque = (pd_target - dof_pos) * stiffness - dof_vel * damping
        over_torque = np.abs(would_be_torque) > torque_limits
        if over_torque.any():
            details = ", ".join(
                f"{self._joint_names[i]}={would_be_torque[i]:.1f}Nm (limit {torque_limits[i]:.0f}Nm)"
                for i in np.where(over_torque)[0]
            )
            issues.append(f"would-be commanded torque exceeds torque_limits: {details}")

        for i, rng in enumerate(self._joint_ranges):
            if rng is None:
                continue
            lo, hi = rng
            if pd_target[i] < lo or pd_target[i] > hi:
                issues.append(
                    f"commanded pd_target for '{self._joint_names[i]}' ({pd_target[i]:.3f} rad) "
                    f"is outside its joint range [{lo:.3f}, {hi:.3f}]"
                )

        if self._prev_raw_action is not None:
            jump = np.abs(raw_action - self._prev_raw_action) > self.ACTION_RATE_WARN_RAD
            if jump.any():
                names = [self._joint_names[i] for i in np.where(jump)[0]]
                issues.append(f"large tick-to-tick action jump (>{self.ACTION_RATE_WARN_RAD} rad) on: {names}")
        self._prev_raw_action = raw_action.copy()

        for issue in issues:
            logger.warning(f"[DRY-RUN][UNSAFE ACTION][tick={self._tick}] {issue}")

        if self._tick % 50 == 0:
            logger.info(
                f"[DRY-RUN][tick={self._tick}] task_mode={self.inner_policy.task_mode} "
                f"lin_vel_cmd={np.round(self.inner_policy.lin_vel_command, 2)} "
                f"ang_vel_cmd={self.inner_policy.ang_vel_command:.2f} "
                f"max|pd_target-dof_pos|={np.max(np.abs(pd_target - dof_pos)):.3f} rad"
            )


def main():
    """Same as scripts/run_pipeline.py, except the sim-mode spawn-pose gap is fixed.

    run_pipeline.py only calls pipeline.prepare() (the ramp-to-default-pose mechanism) if not
    cfg.env.is_sim; its sim-mode fallback is set_default_pose_mode, which only tracker-style
    policies implement. For any other policy (e.g. UnifiedLocoKickPolicy), running that script in
    sim skips both paths, so the control loop starts from whatever pose the MuJoCo model happens
    to spawn in (q=0 for every joint if the XML has no <keyframe>, as holosoma's g1_29dof.xml
    doesn't) instead of the trained default standing pose -- corrupting every dof_pos-relative
    observation from the very first tick.

    The fix here is NOT to call prepare() in sim too: empirically, prepare()'s Phase 1 is a
    multi-second *open-loop* PD ramp with zero policy involvement, and this default pose (a fairly
    deep double-knee crouch) is only *actively* stable -- verified directly by holding it under
    plain PD with no policy running, in sim, which topples the robot in under 2s even from an
    exactly-correct, already-grounded start. The trained policy is continuously correcting balance
    from frame 0 in training; it was never meant to be held via naive fixed-target PD. (This also
    matches holosoma's own reference sim2sim workflow, which supports the robot on a gantry --
    removed only after the policy is already running -- rather than open-loop-holding a pose
    before the policy starts.)

    So in sim, this script instead resets straight to the scene's grounded "default_stand"
    keyframe (see assets/robots/g1/holosoma_model/scene_g1_29dof.xml) and hands control to the
    real policy immediately -- verified empirically to hold a stable stance. On real hardware, the
    robot starts from whatever arbitrary pose a technician left it in (not a known-correct sim
    spawn), so prepare()'s smooth ramp-from-current-pose is still the right (and unaffected)
    mechanism there.

    --dry-run: pipeline.prepare() ramps the robot to its default pose via real torque (env.step()
    calls), independent of any dry_run flag -- so on real hardware, --dry-run also skips prepare()
    entirely, not just the main loop's steps, otherwise "dry run" would still move the robot before
    the loop even starts. Actions are then evaluated against whatever pose the robot happens to be
    in when the script starts.
    """
    args = parse_args()
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()

    robot_pose_redis_client = None
    if args.live_ball:
        if args.ball_source == "redis":
            cfg.ctrl.append(BallPoseRedisCtrlCfg(redis_host=args.redis_host, redis_key=args.ball_redis_key))
            ball_source_desc = f"redis key '{args.ball_redis_key}' on {args.redis_host}"
            perception_cmd = (
                f"python scripts/dummy_ball_perception.py --transport redis --redis-host {args.redis_host} "
                f"--redis-key {args.ball_redis_key} --robot-redis-key {args.robot_redis_key}"
            )
        elif args.ball_source == "ros2":  # see BallPoseRos2Ctrl's module docstring for the two-topic rationale
            cfg.ctrl.append(
                BallPoseRos2CtrlCfg(
                    node_name=args.ros2_node_name,
                    ball_topic=args.ball_topic,
                    aim_topic=args.aim_topic,
                    domain_id=args.ros2_domain_id,
                )
            )
            ball_source_desc = f"ROS2 topics '{args.ball_topic}' (ball) / '{args.aim_topic}' (aim)"
            perception_cmd = (
                f"python scripts/dummy_ball_perception.py --transport ros2 --ball-topic {args.ball_topic} "
                f"--aim-topic {args.aim_topic} --robot-redis-key {args.robot_redis_key}"
            )
        else:  # "udp" -- see BallPoseUdpCtrl's module docstring for why this exists (real hardware
            # where Redis isn't usable and direct cross-distro ROS2 is unverified)
            cfg.ctrl.append(
                BallPoseUdpCtrlCfg(listen_host=args.udp_listen_host, listen_port=args.udp_listen_port)
            )
            ball_source_desc = f"UDP on {args.udp_listen_host}:{args.udp_listen_port}"
            perception_cmd = (
                f"python scripts/foxy_ros2_ball_bridge.py --dest-host <this-host> "
                f"--dest-port {args.udp_listen_port}   (real hardware, run under native Foxy) -- or "
                f"python scripts/dummy_ball_perception.py --transport udp --udp-dest-port "
                f"{args.udp_listen_port}   (sim testing)"
            )

        if cfg.env.is_sim:
            cfg.env.xml = HOLOSOMA_SCENE_WITH_BALL
            # The robot/true-ball pose relay below is a SIM-ONLY testing convenience (letting
            # dummy_ball_perception.py compute a robot-frame reading without any real hardware) --
            # it has no counterpart on real hardware regardless of --ball-source (a real detector is
            # rigidly mounted and reports robot-frame for free), so it always uses Redis, independent
            # of which transport carries the actual kick_ball_pos_b/kick_target_pos_b channel.
            robot_pose_redis_client = redis.Redis(host=args.redis_host, port=6379, db=0)
            logger.warning(
                f"--live-ball (source={args.ball_source}): spawning a physical ball in the MuJoCo scene "
                f"({HOLOSOMA_SCENE_WITH_BALL}), reading kick_ball_pos_b/kick_target_pos_b live from "
                f"{ball_source_desc}, and publishing this process's own robot pose PLUS the ball's true "
                f"world position to redis key '{args.robot_redis_key}' on {args.redis_host} every tick "
                f"(this sim-only relay always uses Redis, regardless of --ball-source). Start the "
                f"perception stream in another terminal now if you haven't -- e.g. '{perception_cmd}' -- "
                f"pipeline construction below will block waiting for its first ball reading."
            )
        else:
            logger.warning(
                f"--live-ball (source={args.ball_source}): no MuJoCo scene swap or robot-pose publish on "
                f"real hardware (dynamics are the physical robot, and a real camera-based detector "
                f"already reports relative to itself without needing this) -- reading "
                f"kick_ball_pos_b/kick_target_pos_b live from {ball_source_desc}. Point your real "
                f"perception process at the same channel/shape (see scripts/dummy_ball_perception.py, "
                f"or for --ball-source udp, scripts/foxy_ros2_ball_bridge.py)."
            )

    pipeline_type = cfg.pipeline_type

    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    ball_ctrl_inst = None
    ball_ctrl_type = None
    if args.live_ball and args.ball_log_hz > 0:
        ball_ctrl_type = _BALL_CTRL_TYPE_BY_SOURCE[args.ball_source]
        if ball_ctrl_type in pipeline.ctrl_manager.controllers:
            ball_ctrl_inst = pipeline.ctrl_manager.controllers[ball_ctrl_type].inst
        else:
            logger.warning(f"[live-ball] expected controller {ball_ctrl_type!r} not found -- --ball-log-hz logging disabled.")

    if args.dry_run:
        logger.warning("=" * 78)
        logger.warning("DRY RUN: policy inference runs every tick, but NO torque will be applied.")
        logger.warning("The robot/sim will NOT move. Unsafe-looking actions are logged as warnings.")
        logger.warning("=" * 78)

    ball_qpos_addr = None
    if cfg.env.is_sim:
        env = pipeline.env
        if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "default_stand") < 0:
            raise RuntimeError(
                "sim env has no 'default_stand' keyframe -- add one to the scene XML "
                "(see scene_g1_29dof.xml) before using this launcher."
            )
        mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
        mujoco.mj_forward(env.model, env.data)
        env.update()
        logger.warning(f"Sim: reset to 'default_stand' keyframe, base_z={env.data.qpos[2]:.3f}")

        if args.live_ball:
            # The dummy/real perception process needs the ball's TRUE simulated position, not just
            # the robot's -- a fixed/fabricated ball position drifts arbitrarily far from what's
            # actually visible in the viewer the moment the robot's feet/legs push the physical
            # ball around, which is exactly what makes "walk up to the visible ball" testing
            # meaningless without this. This is still not a "ball-perception shortcut" in the sense
            # that matters for sim2real: on real hardware there IS no true ball position to read
            # (see the else-branch warning below), so a real detector still has to do real
            # perception there -- this only removes the SIM-SIDE gap between what you see and what
            # gets reported, which a real camera wouldn't have in the first place.
            ball_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
            if ball_jid >= 0:
                ball_qpos_addr = int(env.model.jnt_qposadr[ball_jid])
                ball_qvel_addr = int(env.model.jnt_dofadr[ball_jid])
            else:
                ball_qvel_addr = None
                logger.warning("--live-ball: no 'ball_freejoint' in this scene -- can't publish true ball position.")
    elif args.dry_run:
        logger.warning(
            "Real hardware + --dry-run: skipping pipeline.prepare() (it physically ramps the "
            "robot to its default pose via real torque) -- actions will be evaluated against the "
            "robot's CURRENT pose instead."
        )
    else:
        pipeline.prepare()

    safety_monitor = DryRunSafetyMonitor(pipeline) if args.dry_run else None
    ball_respawner = (
        BallRespawner(pipeline, ball_qpos_addr, ball_qvel_addr) if ball_qpos_addr is not None else None
    )

    _robot_pose_publish_failing = False  # throttle: warn once per failure streak, not every tick
    _last_ball_log_t = 0.0

    while True:
        time_start = time.time()
        pipeline.step(dry_run=args.dry_run)
        if safety_monitor is not None:
            safety_monitor.check_and_log()
        if ball_respawner is not None and not args.dry_run:
            # --dry-run applies no torque (env.step() skipped) -- teleporting the ball there would
            # still visibly move it despite "nothing physically moves" being the documented contract.
            ball_respawner.check_and_respawn()

        if ball_ctrl_inst is not None and time_start - _last_ball_log_t >= (1.0 / args.ball_log_hz):
            _last_ball_log_t = time_start
            task_mode = pipeline.policy.policy.task_mode  # unwrap PolicyWrapper -> UnifiedLocoKickPolicy
            reading = _peek_ball_reading(ball_ctrl_inst, ball_ctrl_type)
            if reading is None:
                logger.info(f"[live-ball] task_mode={task_mode} -- no reading received yet from {ball_ctrl_type}")
            else:
                ball_pos_b, target_pos_b, age, valid = reading
                theta_deg = float(target_pos_b[0]) * args.kick_aim_theta_ref_deg
                age_str = f"{age * 1000:.0f}ms" if age is not None else "n/a"
                logger.info(
                    f"[live-ball] task_mode={task_mode:<10} "
                    f"ball_pos_b={_fmt_vec(ball_pos_b, args.ball_log_decimals)}  "
                    f"target_pos_b={_fmt_vec(target_pos_b, args.ball_log_decimals)}  "
                    f"theta_if_aim={theta_deg:+.1f}deg(ref={args.kick_aim_theta_ref_deg:.0f})  "
                    f"age={age_str}  valid={valid}"
                )

        if robot_pose_redis_client is not None:
            # The dummy/real perception process needs the robot's own pose to correctly compute how
            # the ball's position in the robot's frame changes as the robot walks (pure geometry),
            # PLUS -- in sim, when a physical ball is in the scene -- the ball's true current world
            # position, since the robot can physically push/kick it away from wherever it spawned
            # (see ball_qpos_addr's setup above for why this stopped being optional).
            try:
                payload = {
                    "base_pos_w": np.asarray(pipeline.env.base_pos).tolist(),
                    "base_quat_xyzw": np.asarray(pipeline.env.base_quat).tolist(),
                    "t": time_start,
                }
                if ball_qpos_addr is not None:
                    payload["ball_pos_w"] = pipeline.env.data.qpos[ball_qpos_addr : ball_qpos_addr + 3].tolist()
                robot_pose_redis_client.set(args.robot_redis_key, json.dumps(payload))
                _robot_pose_publish_failing = False
            except RedisError as e:
                if not _robot_pose_publish_failing:
                    logger.warning(f"[live-ball] failed to publish robot/ball pose to redis: {e}")
                    _robot_pose_publish_failing = True
        time_end = time.time()
        time_diff = time_end - time_start

        # keep the pipeline running at the desired frequency
        if not cfg.run_fullspeed:
            time_diff = pipeline.dt - time_diff
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                if not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    if time_diff < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        pipeline.env.shutdown()
                        time.sleep(10)
                        break


if __name__ == "__main__":
    main()
