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
UnifiedManager.task_mode_mask()). ball/target terms default to zero (matching a Stage-B checkpoint,
which never saw a nonzero reading during training), but become live when a BallPoseRedisCtrl is
wired into the pipeline's controller list (see run_pipeline_prepared.py's --live-ball flag and
scripts/dummy_ball_perception.py) — gated to kick mode only, same as every other kick_* term.

Multi-skill motion selection: `kick_motion_command`/`kick_motion_ref_ori_b` are read from the
ONNX's own embedded motion buffer (its `joint_pos`/`joint_vel`/`ref_pos_xyz`/`ref_quat_xyzw`
outputs, indexed by a `time_step` input -- see holosoma's `_OnnxMotionPolicyExporter`). If the
policy was trained with holosoma's N-skill mechanism (`HOLOSOMA_SKILLS_CONFIG`), that buffer holds
ALL configured skills' clips concatenated together, and the ONNX carries
`skill_motion_start_idx`/`skill_motion_end_idx` metadata (one entry per skill) marking each
skill's segment. `[TRIGGER_KICK]` triggers skill 0 (backward compatible); `[TRIGGER_KICK:N]`
triggers skill N, offsetting `time_step` to that skill's own start and capping it at that skill's
own end so it can never drift into a different skill's embedded frames. Checkpoints exported
before this mechanism existed have no such metadata; they behave exactly as before (skill 0 only,
uncapped except by the ONNX's own global clamp).

Per-skill ball/target: `skill_ball_xy`/`skill_target_xy` metadata (same export path) gives each
skill's own configured nominal ball-spawn and shot-target (x, y) -- `get_skill_ball_xy`/
`get_skill_target_xy` below. Deployment scenes/callers decide whether/how to use these (this
policy class itself never spawns or moves anything physical); see mujoco_kick_rollout_worker.py
for the reference consumer.

Selecting which skill to kick, interactively: `[CYCLE_KICK_SKILL]` advances a *pending* selection
(`self._selected_skill_id`, independent of `self.kick_skill_id` which only means anything while
actually kicking) through `0, 1, ..., N-1, 0, ...`; plain `[TRIGGER_KICK]` then kicks whichever
skill is currently pending, instead of always skill 0. `[TRIGGER_KICK:N]` is unaffected -- it's
still an explicit override that also updates the pending selection to N, so a later plain
`[TRIGGER_KICK]` repeats the same skill. This is the mechanism behind the deploy keyboard's "J
cycles skill, K kicks" / joystick's equivalent combo (see g1_unified_loco_kick_cfg.py).
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

        # Per-skill motion-buffer boundaries (see this module's docstring). Absent on checkpoints
        # exported before this mechanism existed -- default to "one skill, starting at 0, with no
        # per-skill upper bound" so old ONNX files behave exactly as they always have.
        if "skill_motion_start_idx" in meta and "skill_motion_end_idx" in meta:
            import json

            self._skill_start_idx: list[int] = [int(v) for v in json.loads(meta["skill_motion_start_idx"])]
            self._skill_end_idx: list[int] = [int(v) for v in json.loads(meta["skill_motion_end_idx"])]
        else:
            self._skill_start_idx = [0]
            self._skill_end_idx = []  # empty -> no per-skill cap, only the ONNX's own global clamp

        # Per-skill "whole authored/raw clip ends here" frame index (same buffer-index space as
        # the two above) -- approach + strike + the actor's own captured post-kick follow-through,
        # BEFORE the synthetic recovery-transition + static-hold tail appended after it. The exact
        # instant a kick_recovery_locomotion_flip_enabled (Stage D) checkpoint was trained to have
        # its task_mode flipped kick->locomotion at. Absent on checkpoints exported before this
        # metadata existed (or any non-Stage-D checkpoint) -- post_step_callback falls back to the
        # older "embedded clip has fully plateaued" heuristic in that case, exactly as before this
        # mechanism existed. See holosoma's get_skill_pre_recovery_metadata (inference_helpers.py).
        if "skill_pre_recovery_motion_end_idx" in meta:
            import json

            self._pre_recovery_idx: list[int] = [int(v) for v in json.loads(meta["skill_pre_recovery_motion_end_idx"])]
        else:
            self._pre_recovery_idx = []

        # Whether this checkpoint was trained with kick_recovery_locomotion_flip_enabled (Stage
        # D's post-swing -> locomotion handoff). When true, post_step_callback auto-returns to
        # locomotion the instant curr_motion_timestep reaches self._pre_recovery_idx above --
        # instead of the older "embedded clip has fully plateaued" heuristic. Gated on this flag
        # (not just "pre_recovery metadata present") because that metadata is exported for
        # every kick-capable checkpoint, Stage D or not -- auto-returning for a checkpoint never
        # trained to expect an early flip would be a real behavior regression. Absent on
        # checkpoints exported before this metadata existed -- defaults False, falling back to the
        # plateau heuristic unchanged. See holosoma's get_kick_recovery_locomotion_flip_metadata
        # (inference_helpers.py).
        if "kick_recovery_locomotion_flip_enabled" in meta:
            import json

            self._kick_recovery_locomotion_flip_enabled: bool = bool(
                json.loads(meta["kick_recovery_locomotion_flip_enabled"])
            )
        else:
            self._kick_recovery_locomotion_flip_enabled = False

        # Per-skill nominal ball spawn / shot-target (x, y), env-local == world for this pipeline's
        # single fixed robot spawn (see get_skill_motion_boundaries_metadata's docstring). Absent
        # on older checkpoints, or ones trained without a ball at all -- callers fall back to their
        # own hardcoded defaults via get_skill_ball_xy/get_skill_target_xy returning None.
        if "skill_ball_xy" in meta and "skill_target_xy" in meta:
            import json

            self._skill_ball_xy: list[list[float]] = json.loads(meta["skill_ball_xy"])
            self._skill_target_xy: list[list[float]] = json.loads(meta["skill_target_xy"])
        else:
            self._skill_ball_xy = []
            self._skill_target_xy = []

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

        # ---- command rate limit: ASYMMETRIC (accel ramped, decel even GENTLER + tiny snap) ----
        # Acceleration is ramped (command_ramp_time) to avoid the jerky full-magnitude step a key
        # press would otherwise command. Deceleration is slower still (command_decel_time):
        # stopping from a fast walk is the hard transient for the policy, and a longer decel
        # walks it down through slower, fully in-distribution speeds before entering standing.
        # Phase-swept stop-from-0.8 falls (v6 Stage-B checkpoint, MuJoCo): instant cut 8/10,
        # 0.15s decel 8/10, 0.5s 3/10, 1.0s 0/10 -- see policy_cfgs.py's field comment for the
        # full experiment (including the disproven fast-decel hypothesis). The tiny snap band
        # just ensures the tail of the ramp cleanly crosses zero_cmd_eps into standing.
        axis_max_mag = np.array([np.abs(m).max() for m in self.commands_map])
        ramp_time = max(cfg_policy.command_ramp_time, 1e-6)
        decel_time = max(cfg_policy.command_decel_time, 1e-6)
        self._cmd_rate_limit_per_tick = (axis_max_mag / ramp_time) / self.freq  # accel, [lin_x, lin_y, ang_z]
        self._cmd_decel_limit_per_tick = (axis_max_mag / decel_time) / self.freq
        self._cmd_zero_snap = cfg_policy.command_zero_snap

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
        self.kick_skill_id = 0
        # pending skill selection: which skill a plain [TRIGGER_KICK] will kick next. Persists
        # across loco<->kick transitions (NOT reset in _return_to_loco, unlike kick_skill_id itself,
        # which is meaningless outside an active kick) so cycling with the robot standing still
        # actually sticks until the next kick.
        self._selected_skill_id = 0
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

    def _query_ref_quat_at(self, time_step: int) -> np.ndarray:
        """Run the ONNX's embedded motion-buffer lookup for ref_quat_xyzw at an arbitrary frame
        index (the `obs` input doesn't affect this output -- see _OnnxMotionPolicyExporter.forward,
        which computes it purely from `time_step`). Used by _trigger_kick to capture the correct
        per-SKILL starting orientation for yaw-offset removal -- self.ref_quat_xyzw_0 only ever
        holds skill 0's own frame-0 value (captured once, at __init__), which is wrong for any
        other skill's own start frame."""
        obs0 = np.zeros(self.session.get_inputs()[0].shape[1], dtype=np.float32)
        outs = self.session.run(
            ["ref_quat_xyzw"],
            {"obs": obs0[None, :], "time_step": np.array([[float(time_step)]], dtype=np.float32)},
        )
        return np.asarray(outs[0]).squeeze()

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

        # Per-axis asymmetric rate limit: accelerating (|target| growing) uses the slow ramp,
        # decelerating uses the fast decel limit, and once the target is zero and the smoothed
        # value is inside the snap band, it snaps straight to zero -- see __init__'s comment for
        # why the slow-glide-to-zero was empirically destabilizing.
        accelerating = np.abs(cmd) > np.abs(self._smoothed_cmd)
        limit = np.where(accelerating, self._cmd_rate_limit_per_tick, self._cmd_decel_limit_per_tick)
        delta = np.clip(cmd - self._smoothed_cmd, -limit, limit)
        self._smoothed_cmd = self._smoothed_cmd + delta
        snap = (cmd == 0.0) & (np.abs(self._smoothed_cmd) < self._cmd_zero_snap)
        self._smoothed_cmd = np.where(snap, 0.0, self._smoothed_cmd)
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
        ball_pos_b: np.ndarray | None = None,
        target_pos_b: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pure obs assembler (no I/O) — takes already-extracted env quantities + current internal
        state (self.task_mode, self.last_action, self.lin_vel_command, self.ang_vel_command,
        self.phase, self.motion_command_t, self.ref_quat_xyzw_t, yaw offsets) and returns the
        261-dim vector in sorted-term order with per-term scales applied. Verified against golden.

        ball_pos_b/target_pos_b: live readings from get_observation's BallPoseRedisCtrl lookup, or
        None if no such controller is wired in (the default) or its latest reading is stale --
        either way falls back to zero, exactly the old hardcoded behavior. Only used in kick mode;
        discarded in loco mode regardless, same as every other kick_* term."""
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

        # kick_* (zeroed in loco mode); ball/target live if a BallPoseRedisCtrl supplied a fresh
        # reading (see _assemble_obs's docstring), else zero -- same as loco mode always is.
        if is_kick:
            kick_motion_command = self.motion_command_t
            kick_motion_ref_ori_b = self._kick_motion_ref_ori_b(torso_quat_xyzw)
            kick_base_ang_vel = base_ang_vel
            kick_dof_pos = dof_pos_rel
            kick_dof_vel = dof_vel
            kick_actions = self.last_action
            kick_ball_pos_b = ball_pos_b if ball_pos_b is not None else Z(3)
            kick_target_pos_b = target_pos_b if target_pos_b is not None else Z(2)
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

    # Both transports implement the identical {"kick_ball_pos_b", "kick_target_pos_b", "valid"}
    # contract (see ball_pose_redis_ctrl.py / ball_pose_ros2_ctrl.py) -- only one is ever wired into
    # a given pipeline's controller list (run_pipeline_prepared.py's --ball-source), so at most one
    # of these keys is present in ctrl_data at a time.
    _LIVE_BALL_CTRL_TYPES = ("BallPoseRedisCtrl", "BallPoseRos2Ctrl", "BallPoseUdpCtrl")

    def _get_live_ball_obs(self, ctrl_data: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Pull the latest kick_ball_pos_b/kick_target_pos_b from whichever live-ball controller
        (Redis or ROS2) is wired into this pipeline's controller list (see run_pipeline_prepared.py's
        --live-ball/--ball-source flags); (None, None) if none is present, or its reading is
        stale/missing, so _assemble_obs falls back to zero exactly as it always has for every other
        config."""
        ball = None
        for ctrl_type in self._LIVE_BALL_CTRL_TYPES:
            ball = ctrl_data.get(ctrl_type)
            if ball is not None:
                break
        if ball is None or not ball.get("valid", False):
            return None, None
        return ball["kick_ball_pos_b"], ball["kick_target_pos_b"]

    def get_observation(self, env_data, ctrl_data):
        self._update_velocity_command(ctrl_data)
        self._update_phase()
        ball_pos_b, target_pos_b = self._get_live_ball_obs(ctrl_data)

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

        obs = self._assemble_obs(
            dof_pos_rel, dof_vel, base_ang_vel, projected_gravity, torso_quat_xyzw, ball_pos_b, target_pos_b
        )
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
            if command == "[TRIGGER_KICK]":
                self._trigger_kick(skill_id=self._selected_skill_id)
            elif command.startswith("[TRIGGER_KICK:") and command.endswith("]"):
                # "[TRIGGER_KICK:N]" -- select which of the ONNX's embedded skills to kick (see
                # this module's docstring). Falls back to skill 0 with a warning if the requested
                # id is out of range, rather than indexing into another skill's segment by
                # accident or raising and killing the whole control loop over a bad command.
                raw = command[len("[TRIGGER_KICK:") : -1]
                try:
                    skill_id = int(raw)
                except ValueError:
                    logger.warning(f"[UnifiedLocoKick] malformed trigger command {command!r}, defaulting to skill 0")
                    skill_id = 0
                if skill_id < 0 or skill_id >= len(self._skill_start_idx):
                    logger.warning(
                        f"[UnifiedLocoKick] requested skill_id={skill_id} out of range "
                        f"(0..{len(self._skill_start_idx) - 1}), defaulting to skill 0"
                    )
                    skill_id = 0
                self._trigger_kick(skill_id=skill_id)
            elif command == "[CYCLE_KICK_SKILL]":
                # Advances the PENDING selection only -- never kicks, never touches kick_skill_id
                # (which stays meaningless/stale until the next actual trigger). Safe to press
                # while already kicking (e.g. accidental press mid-clip): it just changes what the
                # NEXT plain [TRIGGER_KICK] will do, current motion is untouched.
                num_skills = len(self._skill_start_idx)
                self._selected_skill_id = (self._selected_skill_id + 1) % num_skills
                logger.info(f"[UnifiedLocoKick] pending kick skill -> {self._selected_skill_id}")
            elif command == "[RETURN_TO_LOCO]":
                self._return_to_loco()

        if self.task_mode == _TASK_KICK and self.motion_clip_progressing:
            if self._kick_recovery_locomotion_flip_enabled and self._pre_recovery_idx:
                # Stage D (kick_recovery_locomotion_flip_enabled) checkpoint: return to locomotion
                # the instant the WHOLE authored/raw clip ends -- approach + strike + the actor's
                # own captured post-kick follow-through -- the SAME crossing training flips
                # task_mode at, BEFORE the appended synthetic recovery-transition + hold tail.
                # Gated on the flag, not just "boundary metadata present", since this metadata is
                # exported for every kick-capable checkpoint regardless of Stage D. Supersedes the
                # plateau heuristic below entirely for a checkpoint with this flag set: the
                # crossing always happens earlier in clip-time, so the plateau check would never
                # get a chance to fire first anyway.
                if self.curr_motion_timestep >= self._pre_recovery_idx[self.kick_skill_id]:
                    logger.info("[UnifiedLocoKick] authored clip ended (pre_recovery_motion_end_idx reached), auto-returning to locomotion")
                    self._return_to_loco()
            else:
                # auto-return once the embedded clip has clamped at its final (hold) frame long
                # enough (it stops changing tick-to-tick). Guarded by a min-elapsed floor so an
                # early low-motion wind-up segment isn't mistaken for the end. Mirrors holosoma
                # unified.rl_inference. Only reached for checkpoints with the flip flag unset
                # (pre-Stage-D exports, or Stage D disabled) -- identical to this method's behavior
                # before Stage D deployment support existed.
                min_ticks = int(5 * self.freq)
                if self.curr_motion_timestep >= min_ticks and self._prev_motion_command_t is not None:
                    if np.allclose(self.motion_command_t, self._prev_motion_command_t):
                        self._kick_hold_ticks += 1
                    else:
                        self._kick_hold_ticks = 0
                    if self._kick_hold_ticks >= int(3 * self.freq):
                        logger.info("[UnifiedLocoKick] kick clip finished, auto-returning to locomotion")
                        self._return_to_loco()
            # Cap at this skill's own last valid frame so a long-held episode can never drift into
            # the NEXT skill's embedded segment (only possible when boundary metadata is present;
            # older checkpoints fall back to the ONNX's own global time_step_total-1 clamp, exactly
            # as before this mechanism existed).
            if self._skill_end_idx:
                skill_last_frame = self._skill_end_idx[self.kick_skill_id] - 1
                self.curr_motion_timestep = min(self.curr_motion_timestep + 1, skill_last_frame)
            else:
                self.curr_motion_timestep += 1

    def get_skill_ball_xy(self, skill_id: int) -> tuple[float, float] | None:
        """That skill's configured nominal ball spawn (x, y), or None if this checkpoint has no
        such metadata (older export, or trained without a ball) -- callers fall back to their own
        hardcoded default in that case."""
        if 0 <= skill_id < len(self._skill_ball_xy):
            x, y = self._skill_ball_xy[skill_id]
            return float(x), float(y)
        return None

    def get_skill_target_xy(self, skill_id: int) -> tuple[float, float] | None:
        """That skill's configured nominal shot target (x, y), or None -- see get_skill_ball_xy."""
        if 0 <= skill_id < len(self._skill_target_xy):
            x, y = self._skill_target_xy[skill_id]
            return float(x), float(y)
        return None

    def _trigger_kick(self, skill_id: int = 0):
        # capture yaw offsets at the trigger instant: robot from current torso orientation, motion
        # from the SELECTED SKILL's own frame-0 reference (not always self.ref_quat_xyzw_0, which
        # only ever holds skill 0's -- see _query_ref_quat_at's docstring). Uses last-seen torso;
        # refreshed each obs tick anyway.
        self.task_mode = _TASK_KICK
        self.kick_skill_id = skill_id
        self._selected_skill_id = skill_id  # keep pending selection in sync (matters for [TRIGGER_KICK:N])
        start_frame = self._skill_start_idx[skill_id]
        self.curr_motion_timestep = start_frame
        self.motion_clip_progressing = True
        self._kick_hold_ticks = 0
        self.robot_yaw_offset = float(getattr(self, "_last_robot_yaw_cache", 0.0))
        # start_frame == 0 (always true for skill 0) reuses the already-cached value -- byte-
        # identical to this method's pre-multi-skill behavior, zero behavior change for skill 0.
        skill_start_ref_quat = self.ref_quat_xyzw_0 if start_frame == 0 else self._query_ref_quat_at(start_frame)
        self.motion_yaw_offset = float(sRot.from_quat(skill_start_ref_quat).as_euler("xyz")[2])
        logger.info(f"[UnifiedLocoKick] kick triggered (skill_id={skill_id})")

    def _return_to_loco(self):
        self.task_mode = _TASK_LOCOMOTION
        self.motion_clip_progressing = False
        self.curr_motion_timestep = 0
        self.kick_skill_id = 0
        self._kick_hold_ticks = 0
        self.motion_command_t = self.motion_command_0.copy()
        self.ref_quat_xyzw_t = self.ref_quat_xyzw_0.copy()
        self.robot_yaw_offset = 0.0
        self.motion_yaw_offset = 0.0
        logger.info("[UnifiedLocoKick] returned to locomotion")

    def get_init_dof_pos(self) -> np.ndarray:
        return self.default_dof_pos.copy()
