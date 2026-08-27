import logging
import time

import numpy as np
from unitree_cpp import RobotState, SportState, UnitreeController  # type: ignore

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import UnitreeEnvCfg
from robojudo.environment.utils.guard_pose import GUARD_ARM_KD, GUARD_ARM_KP, build_guard_pose
from robojudo.tools.retarget import HandRetarget
from robojudo.utils.rotation import TransformAlignment
from robojudo.utils.util_func import quat_rotate_inverse_np

logger = logging.getLogger(__name__)


@env_registry.register
class UnitreeCppEnv(Environment):
    cfg_env: UnitreeEnvCfg

    # unitree_cpp's own UnitreeController::shutdown() (HansZ8/unitree_cpp, verified from source
    # 2026-08-27 -- see docs/g1_unified_loco_kick_deployment.md and MujocoEnv's identical constant)
    # does `set_gains(kp=[0]*N, kd=[5.0]*N); step([0]*N)`: flat kd=5.0 for every joint, INSTANT (no
    # ramp). soft_stop() below reuses that same damping value so a rehearsed sim run and a real run
    # settle at the same authority, differing only in how gradually kp gets there.
    REAL_ESTOP_DAMPING_KD = 5.0

    # [GUARD_STOP]-only: legs/waist damping while guarding. Deliberately DECOUPLED from
    # REAL_ESTOP_DAMPING_KD above -- that constant exists specifically to replicate real hardware's
    # verified shutdown() value, and changing it would break [SOFT_STOP]'s "exact real replica"
    # property. [GUARD_STOP] has no real-hardware precedent to match at all (it's a reasoned
    # engineering default end to end), so it's free to use a heavier value: 3x
    # REAL_ESTOP_DAMPING_KD, chosen so the legs/waist settle noticeably slower/more resistively
    # (more "damped") than the other stop commands, while still applying ZERO position-holding
    # stiffness there -- still non-active damping mode, just a stronger one. Matches
    # MujocoEnv.GUARD_LEG_DAMPING_KD exactly, so a sim rehearsal predicts real behavior correctly.
    GUARD_LEG_DAMPING_KD = 3.0 * REAL_ESTOP_DAMPING_KD

    def __init__(self, cfg_env: UnitreeEnvCfg, device="cpu"):
        self.enabled: bool = cfg_env.act
        super().__init__(cfg_env=cfg_env, device=device)
        self.RemoteControllerHandler = None
        self._soft_stop_ramp = None  # set by soft_stop(), consumed each step() by _apply_soft_stop_ramp()
        self._guard_ramp = None  # set by guard_stop(), consumed each step() by _apply_guard_ramp()

        cfg_unitree: UnitreeEnvCfg.UnitreeCfg = cfg_env.unitree

        cfg_unitree_dict: dict = cfg_unitree.to_dict()
        cfg_unitree_dict["num_dofs"] = self.num_dofs
        cfg_unitree_dict["stiffness"] = self.stiffness
        cfg_unitree_dict["damping"] = self.damping

        self.robot = cfg_unitree.robot
        self._dof_idx = cfg_env.joint2motor_idx
        self._odometry_type = cfg_env.odometry_type
        if self._odometry_type == "ZED":
            assert self.cfg_env.zed_cfg is not None, "zed_cfg must be set if odometry_type is 'ZED'"
            from robojudo.tools.zed_odometry import ZedOdometry

            self.zed_odometry = ZedOdometry(self.cfg_env.zed_cfg)
        elif self._odometry_type == "DUMMY":
            pass
        elif self._odometry_type == "UNITREE":
            pass

        self.hand_type = cfg_unitree.hand_type
        if self.hand_type == "Inspire":
            self.hand_retarget = HandRetarget(cfg_env.hand_retarget)
        elif self.hand_type == "Dex-3":
            self.hand_retarget = None  # TODO
        else:
            self.hand_retarget = None

        self.sport_state: SportState = None  # pyright: ignore[reportAttributeAccessIssue]
        self.robot_state: RobotState = None  # pyright: ignore[reportAttributeAccessIssue]

        self.unitree = UnitreeController(cfg_unitree_dict)

        # born place alignment extra for h1 torso
        if self.robot == "h1":
            self.torso_align = TransformAlignment()

        # time.sleep(1)  # wait for unitree init
        self.self_check()

    def self_check(self):
        for _ in range(30):
            time.sleep(0.1)
            if self.unitree.self_check():
                logger.info("UnitreeCppEnv self check passed!")
                break
        if not self.unitree.self_check():
            logger.critical("UnitreeCppEnv self check failed!")
            exit()

    def reset(self):
        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
            self.set_born_place()
            self.update()

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

        if self.robot == "h1":
            self.torso_align.set_base(quat=self.torso_quat)

        if self._odometry_type == "ZED":
            self.zed_odometry.set_zreo()

    def update(self):
        # robot state
        self.robot_state = self.unitree.get_robot_state()
        if self._dof_idx is None:
            self._dof_pos = np.array(self.robot_state.motor_state.q, dtype=np.float32)
            self._dof_vel = np.array(self.robot_state.motor_state.dq, dtype=np.float32)
        else:
            self._dof_pos = np.array(
                [self.robot_state.motor_state.q[self._dof_idx[i]] for i in range(len(self._dof_idx))],
                dtype=np.float32,
            )
            self._dof_vel = np.array(
                [self.robot_state.motor_state.dq[self._dof_idx[i]] for i in range(len(self._dof_idx))],
                dtype=np.float32,
            )

        if self.robot == "g1":
            quat = np.array(self.robot_state.imu_state.quaternion, dtype=np.float32)[[1, 2, 3, 0]]
            ang_vel = np.array(self.robot_state.imu_state.gyroscope, dtype=np.float32)
            rpy = np.array(self.robot_state.imu_state.rpy, dtype=np.float32)

            if self.born_place_align:
                quat = self.base_align.align_quat(quat)

            self._base_quat = quat
            self._base_ang_vel = ang_vel
            self._base_rpy = rpy

        elif self.robot == "h1":
            raise NotImplementedError("H1 robot with unitree_cpp not supported yet.")

        # odometry
        if self._odometry_type == "ZED":
            self.zed_odometry.update()
            if self.zed_odometry.is_valid:
                # born place aligned in zed_odometry
                self._base_pos = self.zed_odometry.pos
                self._base_lin_vel = self.zed_odometry.lin_vel
        elif self._odometry_type == "DUMMY":
            self._base_pos = np.array([0.0, 0.0, 0.8])
            self._base_lin_vel = np.array([0.0, 0.0, 0.0])
        elif self._odometry_type == "UNITREE":
            self.sport_state = self.unitree.get_sport_state()
            base_pos = np.asarray(self.sport_state.position, dtype=np.float32)
            lin_vel = np.asarray(self.sport_state.velocity, dtype=np.float32)
            self._base_lin_vel = quat_rotate_inverse_np(self.base_quat, lin_vel)
            self._base_pos = self.base_align.align_pos(base_pos) if self.born_place_align else base_pos

        # FK
        if self.update_with_fk:
            fk_info = self.fk()
            self._torso_pos = fk_info[self._torso_name]["pos"]
            if self.robot != "h1":
                self._torso_quat = fk_info[self._torso_name]["quat"]
                self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]

        # controller
        if self.RemoteControllerHandler:
            self.RemoteControllerHandler(self.robot_state.wireless_remote)

    def soft_stop(self, ramp_seconds: float = 1.5):
        """DELIBERATE, OPT-IN alternative to shutdown() -- ramps kp to 0 over `ramp_seconds` (real
        hardware's own shutdown() is INSTANT, not ramped -- verified from unitree_cpp source, see
        REAL_ESTOP_DAMPING_KD's comment above) while forcing kd to REAL_ESTOP_DAMPING_KD (the same
        flat 5.0 shutdown() itself uses) immediately.

        This is NOT a replacement for [SHUTDOWN] -- that command/method is completely untouched by
        this and remains the fast, unconditional, already-verified-intentional real E-stop (instant
        kp=0/kd=5.0, latched and then continuously re-published by unitree_cpp's own background
        thread even if this Python process later dies -- see UnitreeController's command_writer
        thread). Use [SHUTDOWN] whenever torque needs to come off as fast as possible (e.g. an
        imminent collision, or genuinely unsure what's about to happen). Use soft_stop() only for a
        PLANNED, non-emergency stop where a gentler settle is worth trading away speed for.

        Weaker guarantee than shutdown(): unlike shutdown() (a single set_gains()+step() call, after
        which the C++ side's own background thread keeps re-publishing that latched command forever
        on its own), this ramp is driven from HERE -- it depends on this env's step() continuing to
        be called every control tick for its full duration. If this Python process dies mid-ramp,
        whatever gains were last pushed keep being re-published (same background-thread mechanism),
        which could be a partially-reduced-but-still-nonzero kp frozen in place -- not as safe as
        either a completed ramp or an instant shutdown(). This is exactly why it must never become
        the default/primary E-stop path, only a deliberately-chosen alternative to it.

        No recovery/cancel path is provided (unlike sim's MujocoEnv.estop_recover()) -- once
        triggered, treat it as one-way, matching this class's existing philosophy of having no
        reborn()/recovery mechanism at all (RlPipeline.safety_check() calls shutdown() here, never
        reborn() -- there is no such method on this class). Regaining active policy control after a
        soft_stop() requires a fresh pipeline.prepare() ramp, same as after any other stop.

        NOT VERIFIED AGAINST REAL HARDWARE OR THE REAL unitree_cpp BINDING -- that package isn't
        installed in this checkout (see submodule_cfg.yaml), so this could only be checked by
        reading unitree_cpp's source and unit-testing this method's own ramp math against a mocked
        `unitree` stub, not by an actual run. Verify carefully (gantry-supported, spotter present,
        same caution as any new real-hardware mechanism) before trusting it.
        """
        if self._soft_stop_ramp is None:
            self._soft_stop_start_kp = np.asarray(self.stiffness, dtype=np.float64).copy()
        self._soft_stop_ramp = {
            "start_stiffness": self._soft_stop_start_kp,
            "start_time": time.time(),
            "duration": max(float(ramp_seconds), 1e-6),
        }
        logger.warning(
            f"[SOFT_STOP] Ramping real kp to 0 over {ramp_seconds:.1f}s, kd -> "
            f"{self.REAL_ESTOP_DAMPING_KD:.2f} flat (matches unitree_cpp's own shutdown() damping "
            "value). This is NOT the emergency stop -- use [SHUTDOWN] (A / Esc) if torque needs to "
            "come off immediately."
        )

    def _apply_soft_stop_ramp(self) -> bool:
        """Called once per step(): if soft_stop() is in progress, push interpolated gains to the
        real SDK this tick (unitree_cpp only updates its published/re-broadcast command on a
        set_gains()+step() call -- see this class's step(), which this bypasses while a ramp is
        active). Returns True if it handled this tick's command (caller should skip its normal
        policy-driven step), False if no ramp is active (cheap no-op)."""
        ramp = self._soft_stop_ramp
        if ramp is None:
            return False
        alpha = min((time.time() - ramp["start_time"]) / ramp["duration"], 1.0)
        if self.enabled:
            kp_t = (1.0 - alpha) * ramp["start_stiffness"]
            kd_t = np.full(self.num_dofs, self.REAL_ESTOP_DAMPING_KD, dtype=np.float64)
            self.set_gains(kp_t, kd_t)
            self.unitree.step([0.0] * self.num_dofs)  # zero target -- matches shutdown()'s own convention
        if alpha >= 1.0:
            self._soft_stop_ramp = None
        return True

    def guard_stop(self, leg_ramp_seconds: float = 0.0, arm_ramp_seconds: float = 0.4):
        """DELIBERATE, OPT-IN reflex: legs/waist cut to zero stiffness (flat GUARD_LEG_DAMPING_KD
        damping -- heavier than soft_stop()'s REAL_ESTOP_DAMPING_KD, see that constant's comment
        for why they're deliberately decoupled -- instant by default) WHILE the arms actively
        drive to an overhead head-guard pose -- see robojudo/environment/utils/guard_pose.py for
        the pose's full derivation (FK-verified on the MuJoCo model, NOT validated against real
        hardware -- there's no head body in this G1 model at all, so this targets a generic,
        direction-agnostic failure mode rather than a fall-direction-specific brace).

        Unlike soft_stop() (which only ever reduces authority), this ALSO actively commands the arm
        joints with moderate (GUARD_ARM_KP/KD) stiffness toward a specific target -- the arms move
        with intent, not limply. leg_ramp_seconds defaults to instant (0.0): the premise here is
        "the robot is going down, get the arms up NOW", not a gradual settle -- use soft_stop()
        instead for a no-arm-motion gentle stop. arm_ramp_seconds defaults to 0.4s -- fast enough to
        matter during a fall (this project's own sim2sim measurements put a G1 topple at roughly 1s
        start to finish), slow enough not to be a violent snap near the robot's own head.

        Same weaker-than-[SHUTDOWN] guarantee as soft_stop() (depends on this process's step()
        continuing every tick) and the SAME "no recovery path, one-way" philosophy -- but unlike
        soft_stop(), this NEVER self-clears once triggered (both ramps cap at alpha=1.0 and stay
        there, continuously re-pushing zero leg stiffness and the held arm pose every tick) --
        letting it clear would silently hand arm authority back to whatever the policy's own
        (untrusted, mid-fall) output is, at GUARD_ARM_KP's weakened stiffness, instead of holding
        the guard pose. Regaining active policy control requires a fresh pipeline.prepare().

        NOT VERIFIED AGAINST REAL HARDWARE OR THE REAL unitree_cpp BINDING (not installed in this
        checkout) -- same caution as soft_stop() applies, doubly so here since this is actively
        moving joints near the robot's own head rather than just cutting torque. Watch it happen in
        sim first (MujocoEnv.guard_stop(), identical pose/logic) before ever trusting it on hardware.
        """
        is_arm, target_pose = build_guard_pose(self.dof_cfg.joint_names)
        if not is_arm.any():
            logger.warning(
                "[GUARD_STOP] no arm joints recognized in this robot's DoF config -- nothing to "
                "guard. Falling back to soft_stop()."
            )
            self.soft_stop()
            return
        self._guard_ramp = {
            "is_arm": is_arm,
            "target_pose": target_pose,
            "start_pos": np.asarray(self.dof_pos, dtype=np.float64).copy(),
            "start_leg_kp": np.asarray(self.stiffness, dtype=np.float64).copy(),
            "start_time": time.time(),
            # <=0 means "instant", handled as an explicit alpha=1.0 special case below rather than
            # via a tiny-epsilon division -- see MujocoEnv._apply_guard_ramp's comment for why
            # (found via a mocked-clock test where epsilon-division was NOT guaranteed to reach
            # alpha=1.0 on the very first call, 2026-08-27).
            "leg_duration": float(leg_ramp_seconds),
            "arm_duration": max(float(arm_ramp_seconds), 1e-6),
        }
        logger.warning(
            f"[GUARD_STOP] Legs/waist -> zero stiffness over {leg_ramp_seconds:.1f}s (kd -> "
            f"{self.GUARD_LEG_DAMPING_KD:.1f} flat); arms moving to the overhead guard pose over "
            f"{arm_ramp_seconds:.1f}s and HOLDING there. NOT the emergency stop -- use [SHUTDOWN] "
            "if torque needs to come off immediately."
        )

    def _apply_guard_ramp(self) -> bool:
        """Called once per step(): if guard_stop() is in progress (or held, post-ramp -- see
        guard_stop()'s docstring for why this never self-clears), push this tick's per-joint
        gains+target to the real SDK. Returns True if it handled this tick (caller should skip its
        normal policy-driven step), False if no guard is active (cheap no-op)."""
        ramp = self._guard_ramp
        if ramp is None:
            return False
        is_arm = ramp["is_arm"]
        now = time.time()
        leg_alpha = 1.0 if ramp["leg_duration"] <= 0 else min((now - ramp["start_time"]) / ramp["leg_duration"], 1.0)
        arm_alpha = min((now - ramp["start_time"]) / ramp["arm_duration"], 1.0)

        if self.enabled:
            kp = np.where(is_arm, GUARD_ARM_KP, (1.0 - leg_alpha) * ramp["start_leg_kp"])
            kd = np.where(is_arm, GUARD_ARM_KD, self.GUARD_LEG_DAMPING_KD)
            target = np.where(
                is_arm,
                (1.0 - arm_alpha) * ramp["start_pos"] + arm_alpha * ramp["target_pose"],
                0.0,
            )
            self.set_gains(kp, kd)
            self.unitree.step(target.tolist())
        return True

    def guard_stop_recover(self):
        """SIM-only-style recovery is NOT available here -- see guard_stop()'s docstring (matches
        soft_stop()'s existing one-way philosophy on real hardware). Provided only so calling code
        can check `hasattr(env, "guard_stop_recover")` symmetrically with MujocoEnv without needing
        an env-type branch; calling this on real hardware just logs and does nothing."""
        logger.warning(
            "[GUARD_STOP] No recovery path on real hardware (by design -- see guard_stop()'s "
            "docstring). A fresh pipeline.prepare() is required to regain active control."
        )

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if self._apply_guard_ramp():
            pass
        elif not self._apply_soft_stop_ramp():
            # limits = self.position_limits
            # pd_target_clipped = np.clip(pd_target, limits[:, 0], limits[:, 1])

            # delta = pd_target - pd_target_clipped
            # if np.any(delta != 0):
            #     logger.warning(f"JOINT out of LIMIT-> {delta}")

            # positions = pd_target_clipped
            positions = pd_target
            if self.enabled:
                self.unitree.step(positions.tolist())

        if hand_pose is not None:
            assert type(hand_pose) is np.ndarray, "hand_pose should be a numpy array"
            assert hand_pose.shape[0] == 2, "hand_pose should be of shape (2, -1)"
            if self.hand_retarget is not None:
                hand_pose = self.hand_retarget.from_pose_to_cmd(hand_pose)
                logger.debug(f"Hand pose retargeted: {hand_pose}")
            hand_pose = hand_pose.tolist()

            if self.enabled:
                self.unitree.step_hands(hand_pose[0], hand_pose[1])

    def shutdown(self):
        # self.set_damping_mode()
        self.enabled = False
        self.unitree.shutdown()

    def set_gains(self, stiffness, damping):
        if not hasattr(self, "unitree"):  # TODO
            return
        if not self.enabled:
            return
        # Keep self.stiffness/self.damping in sync with what's actually pushed to the SDK -- was
        # previously only ever set once, from the config default, in Environment.update_dof_cfg()
        # (this method itself never updated them). Harmless for that original call site (it already
        # assigns the same values itself right before calling this), but load-bearing for
        # soft_stop(): without it, self._soft_stop_start_kp would read a STALE value on any 2nd
        # soft_stop() call after a first one already completed (real kp=0 on the robot, but
        # self.stiffness still showing the pre-ramp trained value) -- momentarily re-stiffening an
        # already-limp robot before ramping back down. Found via a soft_stop() test against a fake
        # SDK binding (2026-08-27), fixed before it could matter on real hardware.
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)
        self.unitree.set_gains(stiffness, damping)


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg

    env = UnitreeCppEnv(cfg_env=G1RealEnvCfg())
    env.set_gains(
        stiffness=[kp * 0.0 for kp in env.stiffness],
        damping=[kd * 0.1 for kd in env.damping],
    )
    while 1:
        # env.step(np.zeros(29), np.ones((2, 7)) * -0)
        env.step(np.zeros(29), None)
        # if controller.remote_controller("A"):
        #     controller.shutdown()
        print(env.base_rpy)
        print(env.dof_pos)
        print(env.base_pos)
        env.update()
        # print(env.base_pos)
        time.sleep(0.1)
    # print("Exit")
    # from robojudo.controller import UnitreeCtrl
    # ctrl = UnitreeCtrl(env=env)

    # while True:
    #     env.update()
    #     state = ctrl.get_state()
    #     events = ctrl.get_events()
    #     print("State:", state)
    #     print("Events:", events)
    #     time.sleep(0.1)  # Simulate a control loop
