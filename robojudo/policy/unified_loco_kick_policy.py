"""UnifiedLocoKickPolicy — deploy the holosoma "unified locomotion + ball-kicking" G1 policy
(playground/locomotion_and_ball_kicking, FastSAC, exported ONNX) inside RoboJuDo.

A single policy that does BOTH velocity-command locomotion AND a one-shot motion-clip kick, keyed
by a per-tick ``task_mode``. This is a native RoboJuDo re-implementation of the deploy-side
``holosoma_inference.policies.unified.UnifiedPolicy`` — same 261-dim observation, same ONNX call —
so it runs as a first-class RoboJuDo policy (MuJoCo sim2sim today, UnitreeCppEnv on real G1 next),
with RoboJuDo's controllers, safety checks and born-place alignment.

Observation fidelity is the whole game here: the 261-dim vector must match training exactly or the
policy fails silently. Every term below is verified byte-for-byte against a golden reference dumped
from the known-good ``UnifiedPolicy`` (see tests/ / scratchpad golden_obs.json) before this is
trusted. Convention note: RoboJuDo works in **xyzw** (w-last) quaternions throughout (scipy), while
holosoma uses wxyz internally — the math here is the same rotations expressed in xyzw.

Observation layout (261 dims, ALPHABETICALLY SORTED term names — both training and RoboJuDo sort
before concatenating, so only names/dims/scales matter, not source order):

    kick_actions(29) kick_ball_pos_b(3) kick_base_ang_vel(3) kick_dof_pos(29) kick_dof_vel(29)
    kick_motion_command(58) kick_motion_ref_ori_b(6) kick_target_pos_b(2)
    loco_actions(29) loco_base_ang_vel(3, x0.25) loco_command_ang_vel(1) loco_command_lin_vel(2)
    loco_cos_phase(2) loco_dof_pos(29) loco_dof_vel(29, x0.05) loco_projected_gravity(3)
    loco_sin_phase(2) task_mode_onehot(2)

In locomotion mode every kick_* term is zero; in kick mode every loco_* term is zero (mirrors
UnifiedManager.task_mode_mask()). ball/target terms are held at zero for a Stage-B checkpoint
(training v9 gates them off until shooting_reward_scale>0 — see that repo's README v9).
"""

from __future__ import annotations

import logging

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation as sRot

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import UnifiedLocoKickPolicyCfg
from robojudo.tools.dof import DoFConfig
from robojudo.utils.util_func import command_remap

logger = logging.getLogger(__name__)

_TASK_LOCOMOTION = "locomotion"
_TASK_KICK = "kick"


@policy_registry.register
class UnifiedLocoKickPolicy(Policy):
    cfg_policy: UnifiedLocoKickPolicyCfg

    def __init__(self, cfg_policy: UnifiedLocoKickPolicyCfg, device):
        device = "cpu"
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(cfg_policy.policy_file, ort.SessionOptions(), providers=providers)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        # ---- DoF / gains / action-scale from ONNX modelmeta (self-describing export) ----
        meta = self.session.get_modelmeta().custom_metadata_map

        def _floats(key):
            import json

            return [float(v) for v in json.loads(meta[key])]

        def _strs(key):
            import json

            return [str(v) for v in json.loads(meta[key])]

        dof_names = _strs("dof_names")
        kp = _floats("kp")
        kd = _floats("kd")
        self.per_joint_action_scale = np.asarray(_floats("action_scale"), dtype=np.float64)

        # Default pose is NOT in the ONNX metadata (it lives in the training robot config), so it
        # must be supplied explicitly by the cfg. It is what dof_pos is measured relative to in the
        # observation (dof_pos - default), so a wrong default silently biases every dof_pos term.
        assert cfg_policy.default_dof_pos is not None and len(cfg_policy.default_dof_pos) == len(dof_names), (
            "UnifiedLocoKickPolicyCfg.default_dof_pos must be provided and match the 29 dof_names "
            "(the training robot config's default joint pose)."
        )
        dof_config = DoFConfig(
            joint_names=dof_names,
            default_pos=list(cfg_policy.default_dof_pos),
            stiffness=kp,
            damping=kd,
        )
        cfg_new = cfg_policy.model_copy()
        cfg_new.obs_dof = dof_config
        cfg_new.action_dof = dof_config
        super().__init__(cfg_policy=cfg_new, device=device)

        # ---- gait phase (locomotion) ----
        self.gait_period = cfg_policy.gait_period
        self.phase_dt = 2.0 * np.pi / (self.freq * self.gait_period)
        self.zero_cmd_eps = cfg_policy.zero_cmd_eps

        # ---- command remap (controller -> velocity, within training's [-1,1] range) ----
        self.commands_map = [np.asarray(m, dtype=np.float64) for m in cfg_policy.commands_map]

        # ---- command rate limit (smooth onset/offset instead of an instant per-tick step) ----
        ramp_time = max(cfg_policy.command_ramp_time, 1e-6)
        axis_max_mag = np.array([np.abs(m).max() for m in self.commands_map])
        self._cmd_rate_limit_per_tick = (axis_max_mag / ramp_time) / self.freq  # [lin_x, lin_y, ang_z]

        self.reset()

    # ------------------------------------------------------------------ #
    # lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def reset(self):
        self.task_mode = _TASK_LOCOMOTION
        self.last_action = np.zeros(self.num_actions)
        self.lin_vel_command = np.zeros(2)
        self.ang_vel_command = 0.0
        # rate-limited command actually applied (chases _target_cmd at _cmd_rate_limit_per_tick);
        # lin_vel_command/ang_vel_command above are always derived from this, never set directly.
        self._smoothed_cmd = np.zeros(3)
        # persistent keyboard hold-state for _update_velocity_command (see its docstring): a key
        # counts as "held" from its press event until its matching release event, independent of
        # which ticks those events happen to land on.
        self._wasd_held = {"w": False, "s": False, "a": False, "d": False, "q": False, "e": False}
        # gait phase: left foot at 0, right foot at pi (holosoma init)
        self.phase = np.array([0.0, np.pi])
        self.is_standing = False
        # kick-clip state
        self.curr_motion_timestep = 0
        self.motion_clip_progressing = False
        self._kick_hold_ticks = 0
        self.motion_command_t = np.zeros(2 * self.num_dofs)  # [joint_pos(29), joint_vel(29)]
        self.ref_quat_xyzw_t = np.array([0.0, 0.0, 0.0, 1.0])
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        self._prev_motion_command_t = None
        # warm the ONNX once so a frame-0 clip value exists before the first real obs
        self._prime_clip()

    def _prime_clip(self):
        """Run the ONNX once at time_step 0 with a zero obs to populate the frame-0 clip output,
        mirroring holosoma setup_policy's warmup. Kept separate so `reset()` is cheap/idempotent."""
        obs0 = np.zeros(self.session.get_inputs()[0].shape[1], dtype=np.float32)
        outs = self.session.run(
            ["joint_pos", "joint_vel", "ref_quat_xyzw"],
            {"obs": obs0[None, :], "time_step": np.array([[0.0]], dtype=np.float32)},
        )
        self.motion_command_0 = np.concatenate([outs[0].squeeze(), outs[1].squeeze()])
        self.ref_quat_xyzw_0 = outs[2].squeeze()
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()

    # ------------------------------------------------------------------ #
    # commands from controller (locomotion velocity)                    #
    # ------------------------------------------------------------------ #
    def _update_velocity_command(self, ctrl_data: dict):
        """Read velocity from JoystickCtrl/UnitreeCtrl axes or KeyboardCtrl w/a/s/d/q/e, remapped
        into the training command range. commands_map rows: [lin_x, lin_y, ang_z].

        KeyboardCtrl.get_data() drains an event QUEUE each tick (pynput on_press/on_release
        callbacks) -- "keyboard_event" is only the press/release events that arrived since the
        last poll, not "keys currently held". Reacting only to same-tick events (as an earlier
        version of this method did) means a held key reports a command only on the ticks where an
        OS key-repeat event happens to land (~25Hz on common Linux defaults, after an initial
        ~500ms delay) and ZERO on every tick in between -- a rapid on/off chatter, not a sustained
        command. Empirically confirmed (side-by-side sim) to badly destabilize locomotion: base
        height oscillates/trends down instead of holding steady, forward progress roughly halves,
        lateral drift more than doubles. So w/a/s/d/q/e state is tracked persistently here (a key
        is "held" from its press event until its release event, across ticks) and the command is
        computed from that persistent state every tick, not from same-tick event presence.

        `cmd` below is the instantaneous *target* (full commands_map magnitude the instant a key
        is held, or the raw analog stick value) -- it is then rate-limited into self.lin_vel_command
        / self.ang_vel_command via self._smoothed_cmd (see bottom of this method) so the applied
        command ramps smoothly instead of stepping there in one 20ms tick, matching how holosoma's
        own (gradual, accumulator-based) keyboard scheme behaves -- side-by-side sim testing showed
        an instant step is survivable but visibly rougher than a gradual ramp."""
        cmd = np.zeros(3)  # [lin_x(fwd), lin_y(lateral), ang_z(yaw)]
        for key in ctrl_data:
            if key in ("JoystickCtrl", "UnitreeCtrl"):
                axes = ctrl_data[key]["axes"]
                lx, ly, rx = axes["LeftX"], axes["LeftY"], axes["RightX"]
                cmd[0] = command_remap(ly, self.commands_map[0])
                cmd[1] = command_remap(lx, self.commands_map[1])
                cmd[2] = command_remap(rx, self.commands_map[2])
                break
            if key == "KeyboardCtrl":
                for event in ctrl_data[key]["keyboard_event"]:
                    if event.get("type") != "keyboard" or event["name"] not in self._wasd_held:
                        continue
                    self._wasd_held[event["name"]] = bool(event["pressed"])
                held = self._wasd_held
                raw_fwd = float(held["w"]) - float(held["s"])
                raw_lat = float(held["d"]) - float(held["a"])
                raw_yaw = float(held["q"]) - float(held["e"])
                cmd[0] = command_remap(raw_fwd, self.commands_map[0])
                cmd[1] = command_remap(raw_lat, self.commands_map[1])
                cmd[2] = command_remap(raw_yaw, self.commands_map[2])
                break

        delta = np.clip(cmd - self._smoothed_cmd, -self._cmd_rate_limit_per_tick, self._cmd_rate_limit_per_tick)
        self._smoothed_cmd = self._smoothed_cmd + delta
        self.lin_vel_command = self._smoothed_cmd[:2].copy()
        self.ang_vel_command = float(self._smoothed_cmd[2])

    def _update_phase(self):
        """Advance gait phase; freeze both feet together when commanded velocity ~ 0 (standing).
        Mirrors holosoma unified.UnifiedPolicy.update_phase_time."""
        self.phase = np.fmod(self.phase + self.phase_dt + np.pi, 2 * np.pi) - np.pi
        near_zero = np.linalg.norm(self.lin_vel_command) < self.zero_cmd_eps and abs(self.ang_vel_command) < self.zero_cmd_eps
        if near_zero:
            self.phase = np.pi * np.ones(2)
            self.is_standing = True
        elif self.is_standing:
            self.phase = np.array([0.0, np.pi])
            self.is_standing = False

    # ------------------------------------------------------------------ #
    # observation                                                        #
    # ------------------------------------------------------------------ #
    def _projected_gravity(self, base_quat_xyzw: np.ndarray) -> np.ndarray:
        """Gravity direction expressed in the base frame: R(base)^-1 · [0,0,-1]. Exact analog of
        holosoma quat_rotate_inverse(base_quat, [0,0,-1])."""
        return sRot.from_quat(base_quat_xyzw).inv().apply(np.array([0.0, 0.0, -1.0]))

    def _kick_motion_ref_ori_b(self, torso_quat_xyzw: np.ndarray) -> np.ndarray:
        """6-dim: first two columns of the rotation matrix of the clip's reference-body orientation
        expressed in the robot's reference-body (torso_link) frame, both with their captured yaw
        offsets removed. Mirrors holosoma unified.py get_current_obs_buffer_dict's kick_motion_ref_ori_b:
            motion = remove_yaw(ref_quat_xyzw_t, motion_yaw_offset)
            robot  = remove_yaw(torso_quat,      robot_yaw_offset)
            rel    = robot^-1 * motion ; return rel.as_matrix()[:, :2].flatten()
        remove_yaw(q, yaw) = Rz(-yaw) * R(q)  (pre-multiply), all in xyzw/scipy."""
        motion = sRot.from_euler("z", -self.motion_yaw_offset) * sRot.from_quat(self.ref_quat_xyzw_t)
        robot = sRot.from_euler("z", -self.robot_yaw_offset) * sRot.from_quat(torso_quat_xyzw)
        rel = robot.inv() * motion
        return rel.as_matrix()[:, :2].reshape(-1)

    def _assemble_obs(
        self,
        dof_pos_rel: np.ndarray,
        dof_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        projected_gravity: np.ndarray,
        torso_quat_xyzw: np.ndarray,
    ) -> np.ndarray:
        """Pure obs assembler (no I/O) — takes already-extracted env quantities + current internal
        state (self.task_mode, self.last_action, self.lin_vel_command, self.ang_vel_command,
        self.phase, self.motion_command_t, self.ref_quat_xyzw_t, yaw offsets) and returns the
        261-dim vector in sorted-term order with per-term scales applied. Verified against golden."""
        is_kick = self.task_mode == _TASK_KICK
        Z = np.zeros
        nd = self.num_dofs

        # loco_* (zeroed in kick mode); scales: base_ang_vel 0.25, dof_vel 0.05, else 1.0
        loco_base_ang_vel = (Z(3) if is_kick else base_ang_vel) * 0.25
        loco_projected_gravity = Z(3) if is_kick else projected_gravity
        loco_command_lin_vel = Z(2) if is_kick else self.lin_vel_command
        loco_command_ang_vel = Z(1) if is_kick else np.array([self.ang_vel_command])
        loco_dof_pos = Z(nd) if is_kick else dof_pos_rel
        loco_dof_vel = (Z(nd) if is_kick else dof_vel) * 0.05
        loco_actions = Z(nd) if is_kick else self.last_action
        loco_sin_phase = Z(2) if is_kick else np.sin(self.phase)
        loco_cos_phase = Z(2) if is_kick else np.cos(self.phase)

        # kick_* (zeroed in loco mode); ball/target zero for Stage B (source none == training)
        if is_kick:
            kick_motion_command = self.motion_command_t
            kick_motion_ref_ori_b = self._kick_motion_ref_ori_b(torso_quat_xyzw)
            kick_base_ang_vel = base_ang_vel
            kick_dof_pos = dof_pos_rel
            kick_dof_vel = dof_vel
            kick_actions = self.last_action
        else:
            kick_motion_command = Z(2 * nd)
            kick_motion_ref_ori_b = Z(6)
            kick_base_ang_vel = Z(3)
            kick_dof_pos = Z(nd)
            kick_dof_vel = Z(nd)
            kick_actions = Z(nd)
        kick_ball_pos_b = Z(3)
        kick_target_pos_b = Z(2)

        task_mode_onehot = np.array([0.0, 1.0]) if is_kick else np.array([1.0, 0.0])

        # concatenate in ALPHABETICAL term order (matches training/inference sort)
        obs = np.concatenate(
            [
                kick_actions,
                kick_ball_pos_b,
                kick_base_ang_vel,
                kick_dof_pos,
                kick_dof_vel,
                kick_motion_command,
                kick_motion_ref_ori_b,
                kick_target_pos_b,
                loco_actions,
                loco_base_ang_vel,
                loco_command_ang_vel,
                loco_command_lin_vel,
                loco_cos_phase,
                loco_dof_pos,
                loco_dof_vel,
                loco_projected_gravity,
                loco_sin_phase,
                task_mode_onehot,
            ]
        ).astype(np.float32)
        assert obs.shape[0] == 261, f"assembled obs is {obs.shape[0]}, expected 261"
        return obs

    def get_observation(self, env_data, ctrl_data):
        self._update_velocity_command(ctrl_data)
        self._update_phase()

        dof_pos_rel = env_data.dof_pos - self.default_dof_pos
        dof_vel = env_data.dof_vel
        base_ang_vel = env_data.base_ang_vel
        base_quat_xyzw = env_data.base_quat  # RoboJuDo: w-last
        projected_gravity = self._projected_gravity(base_quat_xyzw)
        # torso_link world orientation for kick_motion_ref_ori_b (holosoma uses pinocchio FK of the
        # same body; RoboJuDo env FK provides it natively, w-last). Fall back to base_quat if the
        # env has no torso FK (only reached in kick mode, where it must be present).
        torso_quat_xyzw = env_data.torso_quat if env_data.torso_quat is not None else base_quat_xyzw

        # cache the torso yaw so post_step_callback's _trigger_kick can capture the robot yaw offset
        # at the exact trigger instant (holosoma _capture_yaw_offsets reads it from the same body).
        self._last_robot_yaw_cache = float(sRot.from_quat(torso_quat_xyzw).as_euler("xyz")[2])

        obs = self._assemble_obs(dof_pos_rel, dof_vel, base_ang_vel, projected_gravity, torso_quat_xyzw)
        extras = {"task_mode": self.task_mode, "timestep": self.curr_motion_timestep}
        return obs, extras

    # ------------------------------------------------------------------ #
    # action (ONNX; override base which assumes torch.jit)              #
    # ------------------------------------------------------------------ #
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        outs = self.session.run(
            ["actions", "joint_pos", "joint_vel", "ref_quat_xyzw"],
            {
                "obs": obs[None, :].astype(np.float32),
                "time_step": np.array([[float(self.curr_motion_timestep)]], dtype=np.float32),
            },
        )
        actions = np.asarray(outs[0]).squeeze()
        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        actions = np.clip(actions, -self.action_clip, self.action_clip) if self.action_clip else actions
        self.last_action = actions.copy()

        # stash this tick's clip frame for the NEXT observation (kick_motion_command / ref ori)
        self._prev_motion_command_t = self.motion_command_t
        self.motion_command_t = np.concatenate([np.asarray(outs[1]).squeeze(), np.asarray(outs[2]).squeeze()])
        self.ref_quat_xyzw_t = np.asarray(outs[3]).squeeze()

        return actions * self.per_joint_action_scale

    # ------------------------------------------------------------------ #
    # commands / kick trigger                                            #
    # ------------------------------------------------------------------ #
    def post_step_callback(self, commands: list[str] | None = None):
        for command in commands or []:
            match command:
                case "[TRIGGER_KICK]":
                    self._trigger_kick()
                case "[RETURN_TO_LOCO]":
                    self._return_to_loco()

        if self.task_mode == _TASK_KICK and self.motion_clip_progressing:
            # auto-return once the embedded clip has clamped at its final (hold) frame long enough
            # (it stops changing tick-to-tick). Guarded by a min-elapsed floor so an early low-motion
            # wind-up segment isn't mistaken for the end. Mirrors holosoma unified.rl_inference.
            min_ticks = int(5 * self.freq)
            if self.curr_motion_timestep >= min_ticks and self._prev_motion_command_t is not None:
                if np.allclose(self.motion_command_t, self._prev_motion_command_t):
                    self._kick_hold_ticks += 1
                else:
                    self._kick_hold_ticks = 0
                if self._kick_hold_ticks >= int(3 * self.freq):
                    logger.info("[UnifiedLocoKick] kick clip finished, auto-returning to locomotion")
                    self._return_to_loco()
            self.curr_motion_timestep += 1

    def _trigger_kick(self):
        # capture yaw offsets at the trigger instant: robot from current torso orientation, motion
        # from the clip's frame-0 reference. Uses last-seen torso; refreshed each obs tick anyway.
        self.task_mode = _TASK_KICK
        self.curr_motion_timestep = 0
        self.motion_clip_progressing = True
        self._kick_hold_ticks = 0
        self.robot_yaw_offset = float(getattr(self, "_last_robot_yaw_cache", 0.0))
        self.motion_yaw_offset = float(sRot.from_quat(self.ref_quat_xyzw_0).as_euler("xyz")[2])
        logger.info("[UnifiedLocoKick] kick triggered")

    def _return_to_loco(self):
        self.task_mode = _TASK_LOCOMOTION
        self.motion_clip_progressing = False
        self.curr_motion_timestep = 0
        self._kick_hold_ticks = 0
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        logger.info("[UnifiedLocoKick] returned to locomotion")

    def get_init_dof_pos(self) -> np.ndarray:
        return self.default_dof_pos.copy()
