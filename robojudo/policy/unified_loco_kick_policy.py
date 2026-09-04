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

Readiness gesture (opt-in, cfg `ready_gesture_enabled` + runtime `[TOGGLE_READY_GESTURE]` key):
while the runtime master switch is ON and a live ball reading's (x, y) stays inside the CURRENTLY
SELECTED skill's trained ball box -- `skill_ball_xy[sel]` +- `(randomize_x, randomize_y)`, the
latter parsed per-skill from the ONNX's `experiment_config` -- the RIGHT arm swings CONTINUOUSLY,
purely as an operator signal ("the ball is where this skill expects it, I'm lined up"). The runtime
switch starts OFF; `[TOGGLE_READY_GESTURE]` (keyboard/joystick, see g1_unified_loco_kick_cfg.py)
flips it, reset() clears it. Amplitude eases in when the ball enters the box and eases out when it
leaves (or the switch is turned off); locomotion-only, standing-only by default; superimposes a
small offset on the right-arm pd_target ONLY, never touching gains / `last_action` / balance / kick.
See `_update_ready_gesture_state` / `_apply_ready_gesture`.

Skill-cycled gesture (opt-in, cfg `skill_cycle_gesture_enabled`): a ONE-SHOT wave of the LEFT arm
every time `[CYCLE_KICK_SKILL]` advances the pending skill selection -- a visual "the cycle press
registered" acknowledgment, and a side cue distinct from the readiness gesture (right arm = ball in
range, left arm = skill cycled). Plays `skill_cycle_gesture_duration_s` of a windowed sine (half-
sine bump envelope x N full periods, so it starts AND ends at exactly zero) on the left
shoulder/elbow pd_target, then stops; re-pressing cycle mid-wave restarts it. Locomotion-only,
pd_target overlay only. No `--live-ball` needed; independent of `ready_gesture_enabled` (opposite
arms). No-op on a single-skill checkpoint. See `_start_skill_cycle_gesture` / `_apply_skill_cycle_gesture`.

Manual kick_aim_theta override (opt-in, cfg `manual_kick_aim_enabled`): `[KICK_AIM_THETA_INC]` /
`[KICK_AIM_THETA_DEC]` (keyboard `.`/`,`, joystick LB+Right/LB+Left) nudge an operator-held
`kick_aim_theta` (degrees) by `manual_kick_aim_step_deg` each press, `[KICK_AIM_THETA_RESET]`
(keyboard `0`, joystick LB+Down) zeros it. While enabled, `kick_target_pos_b` is computed
INTERNALLY every tick as `[kick_aim_theta / kick_aim_theta_ref_deg, 0.0]` -- exactly what
`dummy_ball_perception.py --kick-aim-enabled --kick-aim-theta-deg` publishes, just driven from THIS
process's own controller instead of a second one, and adjustable live instead of fixed for the run.
Clamped to the CURRENTLY SELECTED skill's own trained `kick_aim_theta_max_deg`, parsed per-skill
from the ONNX's `experiment_config` (falls back to the wider `kick_aim_theta_ref_deg` with a warning
if that metadata is absent, or the selected skill wasn't trained `kick_aim_enabled`). Overrides
`kick_target_pos_b` ONLY -- `kick_ball_pos_b` (the ball's own position) is untouched, still live if
`--live-ball` is wired in, zero otherwise. SIGN: positive `kick_aim_theta` = the robot's own LEFT
(holosoma's own atan2 convention -- `config_types/multi_skill.py`'s `resolved_nominal_bearing_deg`
docstring: "0=+x/forward, positive=+y/the robot's own left" -- and `managers/observation/terms/
unified.py`'s `kick_aim_command` passes `kick_aim_theta` straight through with no sign flip), so
`[KICK_AIM_THETA_INC]` (positive delta) swings the kick LEFT, `[KICK_AIM_THETA_DEC]` swings it
RIGHT -- g1_unified_loco_kick_cfg.py's key bindings are chosen so the KEY name matches this AIM
direction, not the raw sign. See `_resolve_ball_and_target` / `_nudge_manual_kick_aim_theta`.

Auto-navigation (opt-in, cfg `autonav_enabled` + runtime `[TOGGLE_AUTONAV]` key, starts OFF): while
ON, `_update_velocity_command` computes its OWN (vx, vy, yaw_rate) instead of reading w/a/s/d/stick
input -- a simple proportional loop closing the live `ball_pos_b` reading (`--live-ball`) onto the
CURRENTLY SELECTED skill's own trained ball box center (same box `_selected_skill_ball_box()`
already uses for the readiness gesture). It ONLY drives locomotion into range; it never triggers the
kick itself (stays a manual `[TRIGGER_KICK]` by design). Commands ZERO (holds position) the instant
`ball_pos_b` lands inside the box, so it can't oscillate past the goal chasing a shrinking residual.
Two things cancel it outright, both same-tick: (1) ANY manual locomotion input -- a held w/a/s/d/q/e
key or joystick deflection past `autonav_manual_deadzone` -- hands control straight back to the
operator; (2) `ball_pos_b` going `None` (no `--live-ball`, stale reading, or the selected skill
having no ball-box metadata) freezes at zero velocity and cancels rather than extrapolating blindly.
Either cancellation clears the runtime switch -- resuming needs an explicit `[TOGGLE_AUTONAV]`
press again, it never silently re-engages.

Control law (signs UNFLIPPED, given this codebase's consistent positive=robot's-own-left convention
-- see the SIGN note on manual kick_aim_theta above). Let `nav_error = ball_pos_b[:2] - box_center`
and `gap` = the euclidean distance from the ball to the box BOUNDARY (0 once the ball is inside):
  - closing speed `= min(kp_approach * gap, max_speed)`, applied along `unit(nav_error)` (aim for the
    centre, not just the edge). Scaling by `gap` rather than `|nav_error|` is what keeps the robot
    from blowing through the small box -- `kp*|nav_error|` keeps commanding ~kp*halfwidth right up
    to the edge and the shared decel rate-limiter lags, so the robot arrives still carrying speed.
  - `yaw_rate = kp_yaw * atan2(nav_error.y, max(ball_pos_b.x, 0.3))` with a ~4 deg deadband, but
    ONLY while `gap > 0.15 m`. The ball's forward distance (not `nav_error.x`) is the denominator
    because `nav_error.x` passes through zero at the standoff and `atan2` then blows up to
    +-90..180 deg for a few-cm lateral error. Near the zone, yaw is dropped entirely -- a residual
    turn there just rotates the ball's apparent position back out.
Each term is clamped to that axis's `commands_map` max magnitude; while auto-nav drives, the shared
rate-limiter's decel is sped up 3x (safe -- auto-nav works at low speed, unlike a full-speed manual
walk). Locomotion-only (kick mode zeroes loco_command_lin_vel/ang_vel downstream anyway -- see
`_assemble_obs`). Commands ZERO (holds) the instant the ball is inside the box. See
`_compute_autonav_cmd`.
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

    # Auto-nav conditioning (see _compute_autonav_cmd): forward-distance floor for the yaw atan2
    # denominator so a ball beside/behind the robot can't produce a divide-by-tiny; a heading
    # deadband so sub-degree lateral residuals don't make the heading hunt near the goal; and a
    # minimum box-gap below which yaw is dropped entirely so vx/vy alone do the final positioning
    # (a residual turn near the zone just drags the ball's apparent position out again). Structural
    # conditioning, not tuning -- the operator-facing knobs are autonav_kp_approach/max_speed/kp_yaw.
    _AUTONAV_YAW_FWD_FLOOR_M = 0.3
    _AUTONAV_YAW_DEADBAND_RAD = float(np.deg2rad(4.0))
    _AUTONAV_YAW_MIN_GAP_M = 0.15
    # Auto-nav decelerates faster than the shared locomotion rate-limiter: it works at deliberately
    # low speeds for precision, so an abrupt stop from ~0.3 m/s is safe (unlike halting a full-speed
    # manual walk, which is what command_decel_time's slow ramp exists to cushion). Applied in
    # _update_velocity_command only while auto-nav is driving.
    _AUTONAV_DECEL_SPEEDUP = 3.0

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

        # ---- ready-gesture: per-skill ball box (skill_ball_xy +- (randomize_x, randomize_y)) ----
        # randomize_x/randomize_y are NOT dedicated metadata keys -- they only live inside the big
        # experiment_config JSON blob, per skill. Parse them here so the gesture can use the exact
        # trained box for each skill; fall back to the config's (x, y) half-widths per skill if the
        # blob is absent or shaped differently (older/other exports).
        n_skills = len(self._skill_start_idx)
        _hw_fallback = list(cfg_policy.ready_gesture_box_halfwidth_fallback_xy)
        self._skill_ball_halfwidth_xy: list[list[float]] = [list(_hw_fallback) for _ in range(n_skills)]
        if "experiment_config" in meta:
            try:
                import json

                _ec = json.loads(meta["experiment_config"])
                _sbc = _ec["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["skill_ball_configs"]
                for _i in range(min(n_skills, len(_sbc))):
                    _rx, _ry = _sbc[_i].get("randomize_x"), _sbc[_i].get("randomize_y")
                    if _rx is not None:
                        self._skill_ball_halfwidth_xy[_i][0] = float(_rx)
                    if _ry is not None:
                        self._skill_ball_halfwidth_xy[_i][1] = float(_ry)
            except (KeyError, TypeError, ValueError, IndexError):
                pass  # keep the per-skill fallback half-widths

        # ---- manual kick_aim_theta override: theta_ref_deg (global) + per-skill theta_max_deg ----
        # Same experiment_config source as the ready-gesture box above, different fields:
        # kick_aim_theta_ref_deg is a single GLOBAL normalization constant (not per-skill);
        # kick_aim_theta_max_deg is the actual TRAINED sampling range, global with an optional
        # per-skill override (None = inherit the global value), and only meaningful for a skill
        # whose own kick_aim_enabled is True. _skill_kick_aim_max_deg[i] is None for any skill that
        # either wasn't trained with kick_aim_enabled, or whose max_deg couldn't be determined --
        # _nudge_manual_kick_aim_theta falls back to +-theta_ref_deg (the widest safe bound) and
        # warns when that happens, exactly mirroring dummy_ball_perception.py's own "getting this
        # right is on the caller" caveat (no ONNX metadata says whether a skill is kick_aim-trained
        # in a way that could be auto-verified beyond what's parsed here).
        self._kick_aim_theta_ref_deg = 45.0  # this project's own stable default; overwritten below if present
        self._skill_kick_aim_max_deg: list[float | None] = [None] * n_skills
        if "experiment_config" in meta:
            try:
                import json

                _ec2 = json.loads(meta["experiment_config"])
                _mc = _ec2["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
                self._kick_aim_theta_ref_deg = float(_mc.get("kick_aim_theta_ref_deg", self._kick_aim_theta_ref_deg))
                _global_max_deg = _mc.get("kick_aim_theta_max_deg")
                _sbc2 = _mc["skill_ball_configs"]
                for _i in range(min(n_skills, len(_sbc2))):
                    if bool(_sbc2[_i].get("kick_aim_enabled", False)):
                        _per_skill_max = _sbc2[_i].get("kick_aim_theta_max_deg")
                        _resolved = _per_skill_max if _per_skill_max is not None else _global_max_deg
                        if _resolved is not None:
                            self._skill_kick_aim_max_deg[_i] = float(_resolved)
            except (KeyError, TypeError, ValueError, IndexError):
                pass  # keep theta_ref_deg's project-default fallback; every skill stays max_deg=None

        self._manual_kick_aim_enabled = bool(cfg_policy.manual_kick_aim_enabled)
        self._manual_kick_aim_step_deg = float(cfg_policy.manual_kick_aim_step_deg)

        self._ready_gesture_enabled = bool(cfg_policy.ready_gesture_enabled)
        self._ready_gesture_ramp_s = float(cfg_policy.ready_gesture_ramp_s)
        self._ready_gesture_shoulder_amp_rad = float(cfg_policy.ready_gesture_shoulder_amp_rad)
        self._ready_gesture_elbow_amp_rad = float(cfg_policy.ready_gesture_elbow_amp_rad)
        self._ready_gesture_freq_hz = float(cfg_policy.ready_gesture_freq_hz)
        self._ready_gesture_only_when_standing = bool(cfg_policy.ready_gesture_only_when_standing)
        try:
            self._right_shoulder_pitch_idx = dof_names.index("right_shoulder_pitch_joint")
            self._right_elbow_idx = dof_names.index("right_elbow_joint")
        except ValueError:
            self._right_shoulder_pitch_idx = self._right_elbow_idx = None
            if self._ready_gesture_enabled:
                logger.warning(
                    "[UnifiedLocoKick] ready_gesture_enabled but right_shoulder_pitch_joint/"
                    "right_elbow_joint are not in dof_names -- readiness gesture disabled."
                )

        # ---- one-shot "skill cycled" LEFT-arm wave (see _start/_apply_skill_cycle_gesture) ----
        self._skill_cycle_gesture_enabled = bool(cfg_policy.skill_cycle_gesture_enabled)
        self._skill_cycle_gesture_duration_s = float(cfg_policy.skill_cycle_gesture_duration_s)
        self._skill_cycle_gesture_shoulder_amp_rad = float(cfg_policy.skill_cycle_gesture_shoulder_amp_rad)
        self._skill_cycle_gesture_elbow_amp_rad = float(cfg_policy.skill_cycle_gesture_elbow_amp_rad)
        self._skill_cycle_gesture_swings = float(cfg_policy.skill_cycle_gesture_swings)
        try:
            self._left_shoulder_pitch_idx = dof_names.index("left_shoulder_pitch_joint")
            self._left_elbow_idx = dof_names.index("left_elbow_joint")
        except ValueError:
            self._left_shoulder_pitch_idx = self._left_elbow_idx = None
            if self._skill_cycle_gesture_enabled:
                logger.warning(
                    "[UnifiedLocoKick] skill_cycle_gesture_enabled but left_shoulder_pitch_joint/"
                    "left_elbow_joint are not in dof_names -- skill-cycled gesture disabled."
                )

        # ---- auto-navigation (see _compute_autonav_cmd) ----
        self._autonav_enabled = bool(cfg_policy.autonav_enabled)
        self._autonav_kp_approach = float(cfg_policy.autonav_kp_approach)
        self._autonav_max_speed = float(cfg_policy.autonav_max_speed)
        self._autonav_kp_yaw = float(cfg_policy.autonav_kp_yaw)
        self._autonav_manual_deadzone = float(cfg_policy.autonav_manual_deadzone)

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
        self._cmd_max_mag = axis_max_mag  # [lin_x, lin_y, ang_z] max magnitude -- reused by autonav's clamp
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
        # readiness-gesture state (see _update_ready_gesture_state / _apply_ready_gesture)
        self._ready_gesture_user_on = False  # runtime master switch, flipped by [TOGGLE_READY_GESTURE]
        self._ready_gesture_engaged = False  # condition currently holds (set every get_observation)
        self._ready_gesture_level = 0.0  # 0..1 amplitude, ramps up while engaged / down while not
        self._ready_gesture_phase = 0.0  # free-running sine phase (rad), reset to 0 once level hits 0
        # one-shot "skill cycled" left-arm wave (see _start/_apply_skill_cycle_gesture)
        self._skill_cycle_gesture_ticks_left = 0  # >0 while the wave is playing; counts down to 0
        self._skill_cycle_gesture_total_ticks = 0  # window length captured when the wave was armed
        # manual kick_aim_theta override (see _nudge/_reset_manual_kick_aim_theta); degrees, held
        # across kicks/returns like _selected_skill_id (an operator dials this in ahead of a kick).
        self._manual_kick_aim_theta_deg = 0.0
        # auto-navigation runtime master switch (see _compute_autonav_cmd), flipped by
        # [TOGGLE_AUTONAV] and auto-cleared by manual input or a lost ball reading. Starts OFF.
        self._autonav_user_on = False
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
    def _update_velocity_command(self, ctrl_data: dict, ball_pos_b: np.ndarray | None):
        """Read velocity from JoystickCtrl/UnitreeCtrl axes or KeyboardCtrl w/a/s/d/q/e, remapped
        into the training command range. commands_map rows: [lin_x, lin_y, ang_z].

        AUTO-NAV (see _compute_autonav_cmd's own docstring for the control law): if the runtime
        switch (self._autonav_user_on) is on, `cmd` below is instead computed from the live
        ball_pos_b reading -- UNLESS this tick's raw manual reading is nonzero, in which case that
        manual input wins immediately and auto-nav is cancelled (self._autonav_user_on -> False)
        right here, same tick. Manual input is checked from the RAW ctrl_data reading (before any
        autonav substitution), never from the possibly-autonav-substituted `cmd` itself -- so
        auto-nav's own commanded motion can never look like "manual input" and self-cancel.

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
        manual_cmd = np.zeros(3)  # [lin_x(fwd), lin_y(lateral), ang_z(yaw)]
        manual_active = False
        for key in ctrl_data:
            if key in ("JoystickCtrl", "UnitreeCtrl"):
                axes = ctrl_data[key]["axes"]
                lx, ly, rx = axes["LeftX"], axes["LeftY"], axes["RightX"]
                manual_cmd[0] = command_remap(ly, self.commands_map[0])
                manual_cmd[1] = command_remap(lx, self.commands_map[1])
                manual_cmd[2] = command_remap(rx, self.commands_map[2])
                manual_active = max(abs(lx), abs(ly), abs(rx)) > self._autonav_manual_deadzone
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
                manual_cmd[0] = command_remap(raw_fwd, self.commands_map[0])
                manual_cmd[1] = command_remap(raw_lat, self.commands_map[1])
                manual_cmd[2] = command_remap(raw_yaw, self.commands_map[2])
                manual_active = any(held.values())
                break

        if manual_active and self._autonav_user_on:
            self._autonav_user_on = False
            logger.info("[UnifiedLocoKick] auto-nav CANCELLED -- manual locomotion input detected")

        cmd = self._compute_autonav_cmd(ball_pos_b) if self._autonav_user_on else manual_cmd

        # Per-axis asymmetric rate limit: accelerating (|target| growing) uses the slow ramp,
        # decelerating uses the fast decel limit, and once the target is zero and the smoothed
        # value is inside the snap band, it snaps straight to zero -- see __init__'s comment for
        # why the slow-glide-to-zero was empirically destabilizing.
        accelerating = np.abs(cmd) > np.abs(self._smoothed_cmd)
        decel_limit = self._cmd_decel_limit_per_tick * (self._AUTONAV_DECEL_SPEEDUP if self._autonav_user_on else 1.0)
        limit = np.where(accelerating, self._cmd_rate_limit_per_tick, decel_limit)
        delta = np.clip(cmd - self._smoothed_cmd, -limit, limit)
        self._smoothed_cmd = self._smoothed_cmd + delta
        snap = (cmd == 0.0) & (np.abs(self._smoothed_cmd) < self._cmd_zero_snap)
        self._smoothed_cmd = np.where(snap, 0.0, self._smoothed_cmd)
        self.lin_vel_command = self._smoothed_cmd[:2].copy()
        self.ang_vel_command = float(self._smoothed_cmd[2])

    def _compute_autonav_cmd(self, ball_pos_b: np.ndarray | None) -> np.ndarray:
        """[lin_x, lin_y, ang_z] target command that homes the robot onto the CURRENTLY SELECTED
        skill's ball box (see this class's module docstring's Auto-navigation section for the full
        control law and its cancellation rules). Called once per tick, ONLY while
        self._autonav_user_on is True (see _update_velocity_command) -- a plain read/compute, this
        method itself is what may flip that switch back off (stale ball / no box), mirroring how
        _update_velocity_command's own manual-input check works.

        Returns np.zeros(3) (hold position) whenever it can't or shouldn't drive: no live ball
        reading, no per-skill box metadata, currently mid-kick, or already inside the box. All of
        the first three ALSO cancel the runtime switch outright (same as a lost ball reading should
        never leave a stale auto-nav silently armed); arriving inside the box does NOT cancel it --
        it just holds at zero, ready to resume driving the instant the ball (or the selected skill,
        which live-changes the target if cycled) moves it back out of range."""
        if self.task_mode != _TASK_LOCOMOTION:
            return np.zeros(3)
        if ball_pos_b is None:
            logger.warning("[UnifiedLocoKick] auto-nav CANCELLED -- ball_pos_b reading lost/stale")
            self._autonav_user_on = False
            return np.zeros(3)
        box = self._selected_skill_ball_box()
        if box is None:
            logger.warning(
                "[UnifiedLocoKick] auto-nav CANCELLED -- selected skill has no ball-box metadata "
                "(no skill_ball_xy on this checkpoint, or the selected id is out of range)"
            )
            self._autonav_user_on = False
            return np.zeros(3)
        (x_lo, x_hi), (y_lo, y_hi) = box
        ball_x, ball_y = float(ball_pos_b[0]), float(ball_pos_b[1])
        if x_lo <= ball_x <= x_hi and y_lo <= ball_y <= y_hi:
            return np.zeros(3)  # arrived -- hold, don't chase a shrinking residual to exactly zero

        target_x, target_y = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
        nav_error = np.array([ball_x - target_x, ball_y - target_y])  # goal-center -> ball, in body frame
        n = float(np.linalg.norm(nav_error))

        # Closing SPEED is governed by how far OUTSIDE the box the ball still is (0 at the boundary),
        # NOT by the distance to the box centre. A plain kp*nav_error keeps commanding ~kp*halfwidth
        # right up to the edge, and with the shared decel rate-limiter the smoothed command lags --
        # so the robot arrives carrying real speed and blows straight through the box. Scaling by
        # the box GAP means the target speed is already ~0 by the time the ball reaches the zone.
        gap_x = max(x_lo - ball_x, ball_x - x_hi, 0.0)
        gap_y = max(y_lo - ball_y, ball_y - y_hi, 0.0)
        gap = float(np.hypot(gap_x, gap_y))
        speed = min(self._autonav_kp_approach * gap, self._autonav_max_speed)
        direction = nav_error / n if n > 1e-6 else np.zeros(2)  # aim for the centre, not just the edge
        vx, vy = speed * direction

        # Yaw: only while there's real ground left to cover (gap above a threshold). Near the zone,
        # a residual heading command just drags the ball's apparent position around (rotation
        # effect) and helps it escape -- let vx/vy do the fine positioning. When it IS applied, the
        # atan2 denominator is the ball's FORWARD distance (robustly positive while approaching), not
        # nav_error.x, which blows up to +-90..180 deg for a few-cm lateral error the moment the
        # robot reaches the standoff. Floor + deadband as before.
        if gap > self._AUTONAV_YAW_MIN_GAP_M:
            fwd_for_yaw = max(ball_x, self._AUTONAV_YAW_FWD_FLOOR_M)
            bearing = float(np.arctan2(nav_error[1], fwd_for_yaw))
            if abs(bearing) < self._AUTONAV_YAW_DEADBAND_RAD:
                bearing = 0.0
            yaw = self._autonav_kp_yaw * bearing
        else:
            yaw = 0.0

        return np.clip(np.array([vx, vy, yaw]), -self._cmd_max_mag, self._cmd_max_mag)

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

    def _manual_kick_aim_target_pos_b(self) -> np.ndarray:
        """The bounded aim command [kick_aim_theta_deg / kick_aim_theta_ref_deg, 0.0] for the
        CURRENT operator-set self._manual_kick_aim_theta_deg -- exactly what dummy_ball_perception.py
        --kick-aim-enabled publishes from its own fixed CLI value, computed here instead from a
        value the controller can adjust live. See manual_kick_aim_enabled's cfg docstring."""
        return np.array([self._manual_kick_aim_theta_deg / self._kick_aim_theta_ref_deg, 0.0], dtype=np.float32)

    def _nudge_manual_kick_aim_theta(self, delta_deg: float):
        """Adjust the operator-set manual kick_aim_theta by delta_deg, clamped to the CURRENTLY
        SELECTED skill's own trained kick_aim_theta_max_deg (see __init__'s experiment_config
        parsing). Falls back to +-kick_aim_theta_ref_deg (the widest value that still stays inside
        the bounded obs slot) with a warning if this checkpoint has no such per-skill metadata, or
        the selected skill wasn't trained with kick_aim_enabled at all -- same "verify this before
        trusting it" caveat dummy_ball_perception.py's own docstring carries; nothing here can
        confirm the selected skill actually wants a manual aim command. No-op (warned) if
        manual_kick_aim_enabled is False in the cfg."""
        if not self._manual_kick_aim_enabled:
            logger.warning(
                "[UnifiedLocoKick] kick_aim_theta nudge ignored -- manual_kick_aim_enabled is False "
                "in the policy cfg (feature not active this run)."
            )
            return
        sid = self._selected_skill_id
        max_deg = self._skill_kick_aim_max_deg[sid] if 0 <= sid < len(self._skill_kick_aim_max_deg) else None
        if max_deg is None:
            max_deg = self._kick_aim_theta_ref_deg
            logger.warning(
                f"[UnifiedLocoKick] skill {sid} has no known kick_aim_theta_max_deg (not trained "
                f"kick_aim_enabled, or this checkpoint lacks the metadata) -- clamping to the wider "
                f"+-{max_deg:.1f} deg normalization reference instead of its real trained range; "
                "verify this skill actually expects a manual aim command before trusting it."
            )
        self._manual_kick_aim_theta_deg = float(
            np.clip(self._manual_kick_aim_theta_deg + delta_deg, -max_deg, max_deg)
        )
        logger.info(
            f"[UnifiedLocoKick] manual kick_aim_theta -> {self._manual_kick_aim_theta_deg:+.1f} deg "
            f"(skill {sid}, +-{max_deg:.1f} deg range)"
        )

    def _reset_manual_kick_aim_theta(self):
        """Zero the operator-set manual kick_aim_theta ('aim along the calibrated nominal bearing',
        matching dummy_ball_perception.py --kick-aim-theta-deg's own 0.0 default). No-op (warned) if
        manual_kick_aim_enabled is False in the cfg."""
        if not self._manual_kick_aim_enabled:
            logger.warning(
                "[UnifiedLocoKick] kick_aim_theta reset ignored -- manual_kick_aim_enabled is False "
                "in the policy cfg (feature not active this run)."
            )
            return
        self._manual_kick_aim_theta_deg = 0.0
        logger.info("[UnifiedLocoKick] manual kick_aim_theta -> +0.0 deg (reset)")

    def _resolve_ball_and_target(self, ctrl_data: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        """(kick_ball_pos_b, kick_target_pos_b) for this tick's observation: kick_ball_pos_b is
        always whatever _get_live_ball_obs returns (untouched by the manual aim override -- it's a
        different obs term, the ball's own position, not the aim command). kick_target_pos_b is
        ALSO whatever _get_live_ball_obs returns UNLESS manual_kick_aim_enabled is on, in which case
        it's replaced entirely by _manual_kick_aim_target_pos_b() -- the operator's own dialed-in
        angle takes over from the ball-perception controller's aim reading (if any)."""
        ball_pos_b, target_pos_b = self._get_live_ball_obs(ctrl_data)
        if self._manual_kick_aim_enabled:
            target_pos_b = self._manual_kick_aim_target_pos_b()
        return ball_pos_b, target_pos_b

    def _selected_skill_ball_box(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """((x_lo, x_hi), (y_lo, y_hi)) of the CURRENTLY SELECTED skill's trained ball box --
        skill_ball_xy[sel] +- self._skill_ball_halfwidth_xy[sel]. None if this checkpoint has no
        per-skill ball geometry, or the selected id is out of range."""
        sid = self._selected_skill_id
        if not (0 <= sid < len(self._skill_ball_xy)) or not (0 <= sid < len(self._skill_ball_halfwidth_xy)):
            return None
        cx, cy = self._skill_ball_xy[sid]
        hx, hy = self._skill_ball_halfwidth_xy[sid]
        return ((cx - hx, cx + hx), (cy - hy, cy + hy))

    def _update_ready_gesture_state(self, ball_pos_b: np.ndarray | None):
        """Set self._ready_gesture_engaged: True while ALL of -- the runtime master switch is on
        (_ready_gesture_user_on, flipped by [TOGGLE_READY_GESTURE]); the live ball reading is inside
        the CURRENTLY SELECTED skill's trained box; task_mode is locomotion; and (if
        _ready_gesture_only_when_standing) the commanded velocity is ~0. Called once per
        get_observation. The actual arm motion -- a continuous swing that eases in/out with this
        flag -- is applied in _apply_ready_gesture."""
        if (
            not self._ready_gesture_enabled
            or not self._ready_gesture_user_on
            or self._right_shoulder_pitch_idx is None
        ):
            self._ready_gesture_engaged = False
            return
        box = self._selected_skill_ball_box()
        engaged = (
            ball_pos_b is not None
            and box is not None
            and self.task_mode == _TASK_LOCOMOTION
            and box[0][0] <= float(ball_pos_b[0]) <= box[0][1]
            and box[1][0] <= float(ball_pos_b[1]) <= box[1][1]
        )
        if engaged and self._ready_gesture_only_when_standing:
            standing = (
                np.linalg.norm(self.lin_vel_command) < self.zero_cmd_eps
                and abs(self.ang_vel_command) < self.zero_cmd_eps
            )
            engaged = engaged and standing

        if engaged and not self._ready_gesture_engaged:
            logger.info(
                f"[UnifiedLocoKick] ball is in skill {self._selected_skill_id}'s trained range "
                "-- right-arm readiness gesture (repeats while it stays in range)"
            )
        self._ready_gesture_engaged = engaged

    def _apply_ready_gesture(self, scaled_action: np.ndarray) -> np.ndarray:
        """Superimpose a CONTINUOUS right-arm swing on the already-scaled action (a pd_target
        offset, in radians) while self._ready_gesture_engaged holds -- amplitude eases in over
        _ready_gesture_ramp_s when engaged and eases out over the same time when not, so the swing
        persists/repeats as long as the ball stays in range with no start/stop jerk. Pure output
        overlay -- not stashed into self.last_action, so the policy never 'sees' or compensates for
        it. task_mode leaving locomotion forces it off (eases out) immediately."""
        engaged = self._ready_gesture_engaged and self.task_mode == _TASK_LOCOMOTION
        step = 1.0 / max(self._ready_gesture_ramp_s * self.freq, 1.0)
        self._ready_gesture_level = (
            min(1.0, self._ready_gesture_level + step) if engaged else max(0.0, self._ready_gesture_level - step)
        )
        if self._ready_gesture_level <= 0.0:
            self._ready_gesture_phase = 0.0  # next engagement restarts cleanly from a zero crossing
            return scaled_action
        self._ready_gesture_phase += 2.0 * np.pi * self._ready_gesture_freq_hz / self.freq
        osc = self._ready_gesture_level * np.sin(self._ready_gesture_phase)
        out = np.asarray(scaled_action, dtype=np.float64).copy()
        out[self._right_shoulder_pitch_idx] += self._ready_gesture_shoulder_amp_rad * osc
        out[self._right_elbow_idx] += self._ready_gesture_elbow_amp_rad * osc
        return out

    def _start_skill_cycle_gesture(self):
        """Arm the ONE-SHOT left-arm wave that acknowledges a [CYCLE_KICK_SKILL] press (the actual
        motion is applied in _apply_skill_cycle_gesture). No-op if the feature is disabled, the
        left-arm joints aren't in dof_names, or a kick clip is currently running (the overlay is
        locomotion-only, same rule as the readiness gesture). Re-arming while a wave is still
        playing just restarts it from the top."""
        if not self._skill_cycle_gesture_enabled or self._left_shoulder_pitch_idx is None:
            return
        if self.task_mode != _TASK_LOCOMOTION:
            return
        self._skill_cycle_gesture_total_ticks = max(1, int(round(self._skill_cycle_gesture_duration_s * self.freq)))
        self._skill_cycle_gesture_ticks_left = self._skill_cycle_gesture_total_ticks

    def _apply_skill_cycle_gesture(self, scaled_action: np.ndarray) -> np.ndarray:
        """Superimpose the one-shot LEFT-arm 'skill cycled' wave on the already-scaled action -- a
        pd_target offset (radians) active only for the skill_cycle_gesture_duration_s window after a
        [CYCLE_KICK_SKILL]. Waveform: a half-sine bump envelope (0 -> 1 -> 0) times
        skill_cycle_gesture_swings full sine periods, so it starts AND ends at exactly zero with no
        jerk regardless of the swing count. Pure output overlay -- never stashed into
        self.last_action, so the policy never sees or compensates for it. Drops the overlay
        immediately if task_mode has left locomotion since the wave started (e.g. a kick fired
        during it)."""
        if self._skill_cycle_gesture_ticks_left <= 0 or self._left_shoulder_pitch_idx is None:
            return scaled_action
        if self.task_mode != _TASK_LOCOMOTION:
            self._skill_cycle_gesture_ticks_left = 0
            return scaled_action
        total = max(self._skill_cycle_gesture_total_ticks, 1)
        self._skill_cycle_gesture_ticks_left -= 1  # total-1 on the first tick ... 0 on the last
        progress = 1.0 - self._skill_cycle_gesture_ticks_left / max(total - 1, 1)  # 0.0 -> 1.0 inclusive
        envelope = np.sin(np.pi * progress)  # 0 -> 1 -> 0 across the window
        osc = envelope * np.sin(2.0 * np.pi * self._skill_cycle_gesture_swings * progress)
        out = np.asarray(scaled_action, dtype=np.float64).copy()
        out[self._left_shoulder_pitch_idx] += self._skill_cycle_gesture_shoulder_amp_rad * osc
        out[self._left_elbow_idx] += self._skill_cycle_gesture_elbow_amp_rad * osc
        return out

    def get_observation(self, env_data, ctrl_data):
        # ball_pos_b resolved FIRST -- _update_velocity_command needs it for auto-nav (see its
        # docstring); this reorder is a no-op for everything else, _resolve_ball_and_target is a
        # pure read of ctrl_data/self._manual_kick_aim_theta_deg with no dependency on command state.
        ball_pos_b, target_pos_b = self._resolve_ball_and_target(ctrl_data)
        self._update_velocity_command(ctrl_data, ball_pos_b)
        self._update_phase()
        self._update_ready_gesture_state(ball_pos_b)

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

        # arm overlays: readiness gesture (right arm) then skill-cycled wave (left arm) -- disjoint
        # joints, pure pd_target offsets, neither stashed into self.last_action.
        scaled = actions * self.per_joint_action_scale
        scaled = self._apply_ready_gesture(scaled)
        scaled = self._apply_skill_cycle_gesture(scaled)
        return scaled

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
                # one-shot left-arm "cycle registered" wave -- only when there's actually more than
                # one skill to move between (a single-skill checkpoint's cycle is a no-op).
                if num_skills > 1:
                    self._start_skill_cycle_gesture()
            elif command == "[TOGGLE_READY_GESTURE]":
                # runtime master switch for the readiness gesture (see _update_ready_gesture_state).
                # Only meaningful when ready_gesture_enabled is set in the cfg -- otherwise the
                # feature is compiled out and this just warns. Persists until toggled again or the
                # policy is reset().
                if not self._ready_gesture_enabled:
                    logger.warning(
                        "[UnifiedLocoKick] [TOGGLE_READY_GESTURE] ignored -- ready_gesture_enabled "
                        "is False in the policy cfg (feature not active this run)."
                    )
                else:
                    self._ready_gesture_user_on = not self._ready_gesture_user_on
                    logger.info(
                        f"[UnifiedLocoKick] readiness gesture {'ON' if self._ready_gesture_user_on else 'OFF'} "
                        "(swings only while the ball is in the selected skill's range)"
                    )
            elif command == "[KICK_AIM_THETA_INC]":
                self._nudge_manual_kick_aim_theta(self._manual_kick_aim_step_deg)
            elif command == "[KICK_AIM_THETA_DEC]":
                self._nudge_manual_kick_aim_theta(-self._manual_kick_aim_step_deg)
            elif command == "[KICK_AIM_THETA_RESET]":
                self._reset_manual_kick_aim_theta()
            elif command == "[TOGGLE_AUTONAV]":
                # runtime master switch for auto-navigation (see _compute_autonav_cmd). Only
                # meaningful when autonav_enabled is set in the cfg -- otherwise the feature is
                # compiled out and this just warns, same pattern as [TOGGLE_READY_GESTURE].
                if not self._autonav_enabled:
                    logger.warning(
                        "[UnifiedLocoKick] [TOGGLE_AUTONAV] ignored -- autonav_enabled is False in "
                        "the policy cfg (feature not active this run)."
                    )
                else:
                    self._autonav_user_on = not self._autonav_user_on
                    logger.info(f"[UnifiedLocoKick] auto-nav {'ON' if self._autonav_user_on else 'OFF'}")
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
