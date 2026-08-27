import logging
import time

import mujoco
import mujoco_viewer
import numpy as np

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.guard_pose import GUARD_ARM_KD, GUARD_ARM_KP, build_guard_pose
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        self.viewer = mujoco_viewer.MujocoViewer(
            self.model,
            self.data,
            width=1200,
            height=900,
            hide_menus=True,
            diable_key_callbacks=True,
        )
        self.viewer.cam.distance = 3.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = 180.0
        # self.viewer._paused = True

        if cfg_env.visualize_extras:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading

        self._apply_random_heading()

        self._pre_estop_gains = None  # (stiffness, damping) set by estop(), consumed by estop_recover()/reborn()
        self._estop_ramp = None  # set by estop(ramp_seconds>0), consumed each step() by _apply_estop_ramp()
        self._guard_ramp = None  # set by guard_stop(), consumed each step() by _apply_guard_ramp()

        self.update()  # get initial state

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if not self.random_heading:
            return
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        q = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[3] = c * q[0] - s * q[3]
        self.data.qpos[4] = c * q[1] - s * q[2]
        self.data.qpos[5] = c * q[2] + s * q[1]
        self.data.qpos[6] = c * q[3] + s * q[0]

    def reborn(self, init_qpos=None):
        self.guard_stop_recover()  # no-op unless guard_stop() is currently active
        self.estop_recover()  # no-op unless estop()/soft_stop() is currently active (and not already cleared above)
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    # unitree_cpp's REAL UnitreeController::shutdown() (HansZ8/unitree_cpp, src/unitree_controller.cpp,
    # verified from source 2026-08-27 -- not installed in this checkout, so read from the public repo
    # rather than assumed): `set_gains(kp=[0]*N, kd=[5.0]*N); step(actions=[0]*N)`. A flat kd=5.0 for
    # EVERY joint, not the per-joint trained values -- and instant, not ramped. Confirmed intentional
    # ("1.0.2: Fix: shutdown as damping mode" in that repo's CHANGELOG.md), and NOT a one-shot pulse:
    # a background recurrent thread there re-publishes that latched command on its own timer for the
    # life of the process, independent of further Python-side step() calls, so the DDS command stream
    # never goes silent. There is no separate set_damping_mode() in unitree_cpp's bound API at all
    # (self_check/step/step_hands/set_gains/shutdown/get_robot_state/get_sport_state is the complete
    # list) -- shutdown() IS the damping-mode transition, not a fallback to something gentler.
    ESTOP_REAL_DAMPING_KD = 5.0

    # [GUARD_STOP]-only: legs/waist damping while guarding. Deliberately DECOUPLED from
    # ESTOP_REAL_DAMPING_KD above -- that constant exists specifically to replicate real hardware's
    # verified shutdown() value, and changing it would break [SOFT_STOP]/[ESTOP_REAL]'s "exact real
    # replica" property. [GUARD_STOP] has no real-hardware precedent to match at all (it's a
    # reasoned engineering default end to end), so it's free to use a heavier damping value: 3x
    # ESTOP_REAL_DAMPING_KD, chosen so the legs/waist settle noticeably slower/more resistively
    # (more "damped", less "just drops") than the other stop commands, while still applying ZERO
    # position-holding stiffness there -- this is still non-active damping mode, just a stronger one.
    GUARD_LEG_DAMPING_KD = 3.0 * ESTOP_REAL_DAMPING_KD

    def estop(self, ramp_seconds: float = 0.0, damping: float | np.ndarray | None = None):
        """SIM-ONLY, for OBSERVING failure behavior -- see ESTOP_REAL_DAMPING_KD's comment above for
        exactly what real hardware's UnitreeCppEnv.shutdown() does (verified from unitree_cpp source).

        Drives stiffness (kp) toward zero while damping (kd) is either left untouched (default) or
        switched to `damping` (a scalar broadcast to every joint, or a per-joint array) -- so torque
        becomes (increasingly) just -dof_vel * kd regardless of what the policy commands: no
        position-holding torque, only resistive damping. Far more informative to watch in the viewer
        than a hard kp=kd=0 free-fall: the robot goes slack and collapses/settles under gravity and
        contact instead of just going rigid or twitching.

        ramp_seconds: 0.0 (default) = instant kp cut in one step() -- matches real hardware's
        shutdown(), which is NOT ramped either. > 0 = kp fades LINEARLY to 0 over that many real
        seconds instead (damping switches immediately, only kp ramps), consumed by
        _apply_estop_ramp() every step() call. An instant cut can actually look MORE violent than a
        ramp: at the instant kp hits 0, whatever lean/momentum the controller was still actively
        fighting is suddenly completely unopposed, whereas a ramp gives damping more time to bleed
        off velocity as stiffness fades -- closer to an actual "slow motion" controlled settle than
        a step-function collapse. (This is a SIM exploration knob -- real hardware has no ramped
        path today; changing that would mean patching unitree_cpp's C++ shutdown() itself.)

        damping: None (default) = keep whatever kd is currently active (e.g. this checkpoint's
        trained per-joint values). A scalar or (num_dofs,) array = use that instead while estopped
        -- pass ESTOP_REAL_DAMPING_KD to reproduce real hardware's exact flat kd=5.0 choice.

        The pipeline keeps calling step() with this env exactly as before -- this only changes what
        step() does with the commanded pd_target, it doesn't stop the loop or close the viewer (use
        [SHUTDOWN] for that). Idempotent: calling twice in a row doesn't clobber the stashed original
        gains (a 2nd call mid-ramp restarts the ramp from the CURRENT, already-reduced stiffness).
        Recovered by estop_recover() or the next reborn() (e.g. via [SIM_REBORN]).
        """
        if self._pre_estop_gains is None:
            self._pre_estop_gains = (self.stiffness.copy(), self.damping.copy())

        target_damping = self.damping if damping is None else np.full(self.num_dofs, damping, dtype=np.float64)

        if ramp_seconds <= 0.0:
            self._estop_ramp = None
            self.set_gains(np.zeros(self.num_dofs), target_damping)
            logger.warning(
                f"[ESTOP] Zeroed stiffness (kp) INSTANTLY, kd={target_damping[0]:.2f}(+) -- robot "
                "has no position-holding torque, only joint damping. Watch it settle/collapse in "
                "the viewer. Press the recover key ('`', [SIM_REBORN]) to restore control."
            )
        else:
            self.damping = target_damping  # switches immediately; only stiffness ramps
            self._estop_ramp = {
                "start_stiffness": self.stiffness.copy(),
                "start_time": time.time(),
                "duration": float(ramp_seconds),
            }
            logger.warning(
                f"[ESTOP] Ramping stiffness (kp) to 0 over {ramp_seconds:.1f}s, kd={target_damping[0]:.2f}(+) "
                "-- a gentler, gradual loss of position-holding torque rather than an instant cut. "
                "Press the recover key ('`', [SIM_REBORN]) to restore control at any point."
            )

    def _apply_estop_ramp(self):
        """Called once per step(): if a ramped estop() is in progress, interpolate stiffness toward
        zero based on elapsed wall-clock time. No-op (cheap) once no ramp is active."""
        ramp = self._estop_ramp
        if ramp is None:
            return
        alpha = min((time.time() - ramp["start_time"]) / ramp["duration"], 1.0)
        self.stiffness = (1.0 - alpha) * ramp["start_stiffness"]
        if alpha >= 1.0:
            self._estop_ramp = None

    def soft_stop(self, ramp_seconds: float = 1.5):
        """[SOFT_STOP] sim implementation -- lets you rehearse the EXACT same command/binding
        you'd trigger on real hardware (see UnitreeCppEnv.soft_stop(), which this is the sim
        counterpart of) before ever trying it there. Thin wrapper around estop() using the REAL
        flat damping value (ESTOP_REAL_DAMPING_KD), not this checkpoint's trained kd -- unlike
        [ESTOP_SLOW]/'o', which is a broader trained-kd exploration tool, this is meant to match
        what real hardware would actually settle at. Recoverable via estop_recover()/[SIM_REBORN]
        in sim (real hardware's soft_stop() has no such recovery path -- see its docstring)."""
        self.estop(ramp_seconds=ramp_seconds, damping=self.ESTOP_REAL_DAMPING_KD)

    def estop_recover(self):
        """Undo estop(): restore the stiffness AND damping that were active before it was called
        (cancelling any in-progress ramp). No-op if estop() was never called, or was already
        recovered -- safe to call unconditionally."""
        if self._pre_estop_gains is not None:
            self._estop_ramp = None
            stiffness, damping = self._pre_estop_gains
            self.set_gains(stiffness, damping)
            self._pre_estop_gains = None
            logger.warning("[ESTOP] Recovered -- stiffness and damping restored.")

    def guard_stop(self, leg_ramp_seconds: float = 0.0, arm_ramp_seconds: float = 0.4):
        """[GUARD_STOP]: legs/waist cut to zero stiffness (flat GUARD_LEG_DAMPING_KD damping --
        heavier than [ESTOP_REAL]/soft_stop()'s ESTOP_REAL_DAMPING_KD, deliberately decoupled from
        it since this command has no real-hardware value to replicate, see that constant's comment
        -- instant by default) WHILE the arms actively drive to an overhead head-guard pose (see
        robojudo/environment/utils/guard_pose.py for the pose's
        full derivation and the reasoning behind it, including why it's an FK-verified-but-not-
        hardware-validated engineering default, not a sourced spec).

        Unlike estop()/soft_stop() (which only ever reduce authority), this ALSO actively commands
        the arm joints toward a specific target with moderate (GUARD_ARM_KP/KD) stiffness -- so the
        arms don't just go limp, they move with intent. leg_ramp_seconds defaults to instant (0.0):
        the point of this command is "the robot is going down, get the arms up NOW", not a gentle
        settle -- use [SOFT_STOP] instead if a gradual, no-arm-motion stop is what you want.
        arm_ramp_seconds defaults to 0.4s -- fast enough to matter during a fall (this project's own
        sim2sim measurements put a G1 topple at roughly 1s start to finish), slow enough not to be
        a violent snap.

        Once triggered, the guard state is HELD INDEFINITELY (this method does not self-clear like
        estop()/soft_stop() do) -- both ramps cap at alpha=1.0 and stay there, continuously
        re-asserting zero leg stiffness and the held arm pose every tick, rather than silently
        handing arm authority back to the policy's own (untrusted, mid-fall) output once the ramp
        finishes. Recoverable via guard_stop_recover() or the next reborn() (e.g. [SIM_REBORN]),
        exactly like estop()/soft_stop().
        """
        is_arm, target_pose = build_guard_pose(self.dof_cfg.joint_names)
        if not is_arm.any():
            logger.warning(
                "[GUARD_STOP] no arm joints recognized in this robot's DoF config (joint names "
                "didn't match any known left_/right_ + shoulder/elbow/wrist suffix) -- nothing to "
                "guard. Falling back to a plain instant estop()."
            )
            self.estop(damping=self.ESTOP_REAL_DAMPING_KD)
            return
        if self._pre_estop_gains is None:
            self._pre_estop_gains = (self.stiffness.copy(), self.damping.copy())
        self._guard_ramp = {
            "is_arm": is_arm,
            "target_pose": target_pose,
            "start_pos": self.dof_pos.copy(),
            "start_leg_kp": self.stiffness.copy(),
            "start_time": time.time(),
            # <=0 duration means "instant" -- handled as an explicit alpha=1.0 special case in
            # _apply_guard_ramp, NOT via a tiny-epsilon division. Dividing by an epsilon only
            # reaches alpha=1.0 once real elapsed time exceeds it, which is true in virtually every
            # real tick but is NOT guaranteed on the very first call (e.g. if step() runs unusually
            # fast, or in a test with a mocked/frozen clock) -- found via exactly that mocked-clock
            # scenario in testing (2026-08-27) before it could matter for real.
            "leg_duration": leg_ramp_seconds,
            "arm_duration": max(arm_ramp_seconds, 1e-6),
        }
        logger.warning(
            f"[GUARD_STOP] Legs/waist -> zero stiffness over {leg_ramp_seconds:.1f}s (kd -> "
            f"{self.GUARD_LEG_DAMPING_KD:.1f} flat); arms moving to the overhead guard pose over "
            f"{arm_ramp_seconds:.1f}s and HOLDING there. Press the recover key ('`', [SIM_REBORN]) "
            "to restore control."
        )

    def _apply_guard_ramp(self, pd_target: np.ndarray) -> np.ndarray:
        """Called once per step(), BEFORE the torque computation: if guard_stop() is active,
        overrides this tick's stiffness/damping (in place, same as _apply_estop_ramp) AND returns a
        MODIFIED pd_target -- legs/waist forced to 0 (irrelevant once kp=0 there, kept for
        cleanliness), arms driven toward the held guard pose. Returns pd_target UNCHANGED if no
        guard is active (cheap no-op)."""
        ramp = self._guard_ramp
        if ramp is None:
            return pd_target
        is_arm = ramp["is_arm"]
        now = time.time()
        leg_alpha = 1.0 if ramp["leg_duration"] <= 0 else min((now - ramp["start_time"]) / ramp["leg_duration"], 1.0)
        arm_alpha = min((now - ramp["start_time"]) / ramp["arm_duration"], 1.0)

        kp = self.stiffness.copy()
        kd = self.damping.copy()
        pd = np.asarray(pd_target, dtype=np.float64).copy()

        kp[~is_arm] = (1.0 - leg_alpha) * ramp["start_leg_kp"][~is_arm]
        kd[~is_arm] = self.GUARD_LEG_DAMPING_KD
        pd[~is_arm] = 0.0

        kp[is_arm] = GUARD_ARM_KP
        kd[is_arm] = GUARD_ARM_KD
        pd[is_arm] = (1.0 - arm_alpha) * ramp["start_pos"][is_arm] + arm_alpha * ramp["target_pose"][is_arm]

        self.stiffness = kp
        self.damping = kd
        return pd
        # NOTE: deliberately never clears self._guard_ramp -- see guard_stop()'s docstring for why
        # (letting it clear would silently hand arm authority back to the policy's raw output at
        # GUARD_ARM_KP's weakened stiffness instead of holding the guard pose).

    def guard_stop_recover(self):
        """Undo guard_stop(): restore the stiffness/damping active before it was called and stop
        overriding pd_target. No-op if guard_stop() was never called, or was already recovered."""
        if self._guard_ramp is not None:
            self._guard_ramp = None
            self.estop_recover()  # also restores stiffness/damping via the shared _pre_estop_gains stash
            logger.warning("[GUARD_STOP] Recovered -- stiffness/damping restored, arm override released.")

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        # Explicit bounded slice, not [-num_dofs:]: the robot's freejoint (7 qpos / 6 qvel) is
        # assumed first (see base_pos/quat below, which hardcode qpos[:3]/qpos[3:7]), and its
        # num_dofs joints immediately follow. [-num_dofs:] only happens to agree with that when
        # the robot is the ENTIRE model -- any body added after the robot (e.g. a kickable ball
        # freejoint) silently corrupts dof_pos/dof_vel into a mix of the robot's own tail joints
        # and the extra body's raw qpos/qvel, since the slice keeps grabbing the last N regardless
        # of what's now sitting there. This slice is mathematically identical to [-num_dofs:] for
        # a robot-only model (qpos[7:7+num_dofs] == qpos[-num_dofs:] when nq == 7+num_dofs) and
        # correct for any scene with extra bodies appended after the robot.
        dof_pos = self.data.qpos.astype(np.float32)[7 : 7 + self.num_dofs]
        dof_vel = self.data.qvel.astype(np.float32)[6 : 6 + self.num_dofs]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        quat = self.data.qpos.astype(np.float32)[3:7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[3:6]
        base_pos = self.data.qpos.astype(np.float32)[:3]
        lin_vel = self.data.qvel.astype(np.float32)[0:3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None:
            logger.info("Hand pose-->", hand_pose)

        self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        if self.viewer.is_alive:
            self.viewer.render()

        self._apply_estop_ramp()
        pd_target = self._apply_guard_ramp(pd_target)

        for _ in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)

            self.data.ctrl = torque

            mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
            self.update(simple=True)
        self.update(simple=False)

    def shutdown(self):
        self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
