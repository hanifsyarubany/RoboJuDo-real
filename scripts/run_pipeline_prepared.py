# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
import time

import mujoco
import numpy as np

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

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
    args = parser.parse_args()
    return args


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
        self._joint_ranges = []
        for name in self._joint_names:
            jid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            limited = jid >= 0 and bool(self.env.model.jnt_limited[jid])
            self._joint_ranges.append(tuple(self.env.model.jnt_range[jid]) if limited else None)

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

    pipeline_type = cfg.pipeline_type

    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    if args.dry_run:
        logger.warning("=" * 78)
        logger.warning("DRY RUN: policy inference runs every tick, but NO torque will be applied.")
        logger.warning("The robot/sim will NOT move. Unsafe-looking actions are logged as warnings.")
        logger.warning("=" * 78)

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
    elif args.dry_run:
        logger.warning(
            "Real hardware + --dry-run: skipping pipeline.prepare() (it physically ramps the "
            "robot to its default pose via real torque) -- actions will be evaluated against the "
            "robot's CURRENT pose instead."
        )
    else:
        pipeline.prepare()

    safety_monitor = DryRunSafetyMonitor(pipeline) if args.dry_run else None

    while True:
        time_start = time.time()
        pipeline.step(dry_run=args.dry_run)
        if safety_monitor is not None:
            safety_monitor.check_and_log()
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
