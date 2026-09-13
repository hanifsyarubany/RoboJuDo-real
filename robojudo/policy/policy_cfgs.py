from pydantic import field_validator, model_validator

from robojudo.config import ASSETS_DIR, Config
from robojudo.tools.tool_cfgs import DoFConfig


class PolicyCfg(Config):
    policy_type: str  # name of the policy class
    robot: str  # robot name, e.g. "g1"

    @property
    def policy_file(self) -> str:
        """path to the policy file, to be overrided in subclass"""
        policy_file = ASSETS_DIR / f"models/{self.robot}/PLCAEHOLDER.pt"
        return policy_file.as_posix()

    disable_autoload: bool = False  # if True, disable auto loading of the policy file

    freq: int = 50  # control frequency (Hz)

    obs_dof: DoFConfig
    action_dof: DoFConfig

    # action post processing
    action_scale: float = 1.0
    action_clip: float | None = None  # clip action to [-action_clip, action_clip]
    action_beta: float = 1.0  # action smoothing factor

    # history settings
    history_length: int = 0  # number of history observations to use

    # TODO
    # # upper body override settings
    # wrist_override_idxs: list[int] = []  # indices of the wrist joints to override

    @property
    def history_obs_size(self) -> int:
        """size of the history observations, to be calc in subclass"""
        return 0

    @field_validator("action_scale", "action_clip")
    def check_action_scale(cls, v):
        if v is not None and v <= 0:
            raise ValueError("action_scale must be positive")
        return v

    @model_validator(mode="after")
    def check_history(self):
        if self.history_length < 0:
            raise ValueError("history_length cannot be negative")
        if self.history_obs_size < 0:
            raise ValueError("history_obs_size cannot be negative")
        return self


class UnitreePolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        ang_vel: float = 0.25
        command: list[float] = [2.0, 2.0, 0.25]

    policy_type: str = "UnitreePolicy"
    policy_name: str = "policy"

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/unitree/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.25
    action_clip: float | None = None
    action_beta: float = 0.8

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    max_cmd: list[float] = [0.8, 0.5, 1.57]
    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]


class UnitreeWoGaitPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 0.2
        gravity: float = 1.0
        dof_pos: float = 1.0
        dof_vel: float = 0.05
        command: list[float] = [1.0, 1.0, 1.0]

    policy_type: str = "UnitreeWoGaitPolicy"
    policy_name: str = "policy_wo_gait"

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/unitree/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.25
    action_clip: float | None = None
    action_beta: float = 1.0

    history_length: int = 5  # number of history observations to use
    history_obs_dims: dict[str, int] = {}

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    max_cmd: list[float] = [0.8, 0.5, 1.57]
    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]


class SmoothPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 0.25
        dof_vel: float = 0.05
        lin_vel: float = 0.5

    policy_type: str = "SmoothPolicy"
    policy_name: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/smooth/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.5
    action_clip: float | None = 10.0
    action_beta: float = 0.8

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()

    history_length: int = 10

    @property
    def history_obs_size(self) -> int:
        history_obs_size = 2 + 3 + 3 + 2 + 2 * self.obs_dof.num_dofs + self.action_dof.num_dofs
        return history_obs_size

    cycle_time: float = 0.8

    commands_map: list[list[float]] = [
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
    ]


class H2HPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 1.0
        dof_vel: float = 1.0

    # obs_type as "v-teleop-extend-vr-max-nolinvel"
    policy_type: str = "H2HStudentPolicy"
    policy_name: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/h2h/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.25
    action_clip: float | None = 10.0
    action_beta: float = 0.8

    # ======= POLICY SPECIFIC CONFIGURATION =======
    use_imu_torso: bool = False
    use_dof_pos_offset: bool = False

    obs_scales: ObsScalesCfg = ObsScalesCfg()

    history_length: int = 25

    @property
    def history_obs_size(self) -> int:
        history_obs_size = 2 * self.obs_dof.num_dofs + 3 + 3 + self.action_dof.num_dofs
        return history_obs_size


class AMOPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 0.25
        dof_vel: float = 0.05

    policy_type: str = "AMOPolicy"

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/amo/amo_jit.pt"
        return policy_file.as_posix()

    @property
    def policy_adapter_file(self) -> str:
        policy_adapter_file = ASSETS_DIR / f"models/{self.robot}/amo/adapter_jit.pt"
        return policy_adapter_file.as_posix()

    @property
    def policy_adapter_norm_file(self) -> str:
        policy_adapter_norm_file = ASSETS_DIR / f"models/{self.robot}/amo/adapter_norm_stats.pt"
        return policy_adapter_norm_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()

    action_scale: float = 0.25

    commands_map: list[list[float]]


class BeyondMimicPolicyCfg(PolicyCfg):
    policy_type: str = "BeyondMimicPolicy"
    disable_autoload: bool = True

    policy_name: str
    max_timestep: int = -1
    start_timestep: int = 0

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/beyondmimic/{self.policy_name}.onnx"
        return policy_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    action_scales: list[float]

    without_state_estimator: bool
    override_robot_anchor_pos: bool = True  # if True, drop pos fdb

    use_modelmeta_config: bool = True  # if True, use the config from modelmeta
    use_motion_from_model: bool = True  # if True, use the motion data of onnx model

    @model_validator(mode="after")
    def check_modelmeta(self):
        if self.use_motion_from_model:
            if not self.use_modelmeta_config:
                raise ValueError("use_modelmeta_config must be True when use_motion_from_model")

        return self


class AsapPolicyCfg(PolicyCfg):
    policy_type: str = "AsapPolicy"
    disable_autoload: bool = True

    # ======= MOTION POLICY CONFIGURATION =======
    policy_name: str
    relative_path: str

    motion_length_s: float
    start_upper_body_dof_pos: list[float] | None = None  # reserved for interpolation loco to mimic

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/asap/mimic/{self.policy_name}/{self.relative_path}"
        return policy_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    class ObsScalesCfg(Config):
        # base_lin_vel: float
        base_ang_vel: float
        projected_gravity: float
        # command_lin_vel: float
        # command_ang_vel: float
        # command_stand: float
        # command_base_height: float
        # ref_upper_dof_pos: float
        dof_pos: float
        dof_vel: float
        history: float
        actions: float
        # phase_time: float
        ref_motion_phase: float
        # sin_phase: float
        # cos_phase: float

    action_scale: float = 0.25
    action_clip: float | None = 100.0
    obs_scales: ObsScalesCfg

    history_length: int = 4  # number of history observations to use
    history_obs_dims: dict[str, int] = {}
    """
    Note: the history obs item should be aligned with code of policy
    IMPORTANT: the key order should be SORTED when concat history obs!!!
    """

    USE_HISTORY: bool


class AsapLocoPolicyCfg(PolicyCfg):
    policy_type: str = "AsapLocoPolicy"
    disable_autoload: bool = True

    # ======= MOTION POLICY CONFIGURATION =======
    policy_name: str
    relative_path: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/asap/dec_loco/{self.policy_name}/{self.relative_path}"
        return policy_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    class ObsScalesCfg(Config):
        # base_lin_vel: float
        base_ang_vel: float
        projected_gravity: float
        command_lin_vel: float
        command_ang_vel: float
        command_stand: float
        command_base_height: float
        ref_upper_dof_pos: float
        dof_pos: float
        dof_vel: float
        history: float
        actions: float
        # phase_time: float
        ref_motion_phase: float
        sin_phase: float
        cos_phase: float

    action_scale: float = 0.25
    action_clip: float | None = 100.0
    obs_scales: ObsScalesCfg

    history_length: int = 4  # number of history observations to use
    history_obs_dims: dict[str, int] = {}
    """Note: the history obs item should be aligned with code of policy"""

    USE_HISTORY: bool
    GAIT_PERIOD: float
    NUM_UPPER_BODY_JOINTS: int

    # ======= Default Command CONFIGURATION =======
    command_base_height_default: float


class KungfuBotGeneralPolicyCfg(PolicyCfg):
    policy_type: str = "KungfuBotGeneralPolicy"
    disable_autoload: bool = True

    # ======= MOTION POLICY CONFIGURATION =======
    policy_name: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/kungfubot2/{self.policy_name}.onnx"
        return policy_file.as_posix()

    # ======= POLICY SPECIFIC CONFIGURATION =======
    class ObsScalesCfg(Config):
        # base_lin_vel: float
        base_ang_vel: float
        dof_pos: float
        dof_vel: float
        actions: float
        roll_pitch: float
        # anchor_ref_pos: float
        anchor_ref_rot: float
        next_step_ref_motion: float
        history: float
        future_motion_root_height: float
        future_motion_roll_pitch: float
        future_motion_base_lin_vel: float
        future_motion_base_yaw_vel: float
        future_motion_dof_pos: float

    action_scale: float = 0.0  # not used, scale for each dof
    action_clip: float | None = 100.0
    action_scales: list[float]
    obs_scales: ObsScalesCfg

    history_length: int = 10  # number of history observations to use
    history_obs_dims: dict[str, int] = {}
    """
    Note: the history obs item should be aligned with code of policy
    IMPORTANT: the key order should be SORTED when concat history obs!!!
    """

    compatibility_old_version: bool = False
    """For old version of kungfubot general policy (before 2025-11-13 bugfix #68)"""


class TwistPolicyCfg(PolicyCfg):
    class ObsScalesCfg(Config):
        ang_vel: float = 0.25
        dof_vel: float = 0.05
        dof_pos: float = 1.0

    policy_type: str = "TwistPolicy"
    policy_name: str

    @property
    def policy_file(self) -> str:
        policy_file = ASSETS_DIR / f"models/{self.robot}/twist/{self.policy_name}.pt"
        return policy_file.as_posix()

    action_scale: float = 0.5
    action_clip: float | None = 10.0
    action_beta: float = 1.0

    # ======= POLICY SPECIFIC CONFIGURATION =======
    obs_scales: ObsScalesCfg = ObsScalesCfg()

    history_length: int = 10

    @property
    def n_mimic_obs(self) -> int:
        return self.action_dof.num_dofs + 8

    @property
    def history_obs_size(self) -> int:
        history_obs_size = self.n_mimic_obs + 3 + 2 + 3 * self.action_dof.num_dofs
        return history_obs_size

    ankle_idx: list[int]
    mimic_obs_total_degrees: int
    mimic_obs_wrist_ids: list[int]

    @property
    def mimic_obs_other_ids(self) -> list[int]:
        return [f for f in range(self.mimic_obs_total_degrees) if f not in self.mimic_obs_wrist_ids]


class UnifiedLocoKickPolicyCfg(PolicyCfg):
    """Config for UnifiedLocoKickPolicy — the holosoma unified locomotion+ball-kicking G1 ONNX.

    dof_names / kp / kd / action_scale are read from the ONNX modelmeta at load time (the export is
    self-describing), so they are NOT set here. ``default_dof_pos`` IS required here: the training
    default pose is the frame every dof_pos observation is measured against and is the one thing the
    ONNX metadata does not carry. ``obs_dof`` / ``action_dof`` are required by PolicyCfg but are
    placeholders — the policy overrides them from modelmeta in __init__ (same pattern as
    BeyondMimicPolicy).
    """

    policy_type: str = "UnifiedLocoKickPolicy"
    disable_autoload: bool = True  # ONNX, loaded by the policy, not torch.jit by the base class

    freq: int = 50
    action_beta: float = 1.0  # no action smoothing (holosoma applies none)
    action_clip: float | None = 100.0  # holosoma clips raw actions to [-100, 100] before scaling

    # Absolute path to the exported unified ONNX (lives in the holosoma logs dir, outside RoboJuDo
    # assets), e.g. .../logs/.../unified-stageB-.../model_0119000.onnx
    onnx_path: str

    @property
    def policy_file(self) -> str:
        return self.onnx_path

    # The training robot config's default joint pose (29,), in the ONNX dof_names order. Required.
    default_dof_pos: list[float]

    # gait phase (locomotion): matches training rl_rate(=freq) and gait_period
    gait_period: float = 1.0
    zero_cmd_eps: float = 0.01  # |command| below this => standing (both feet phase locked together)

    # controller-normalized-input -> velocity remap, rows [lin_x(fwd), lin_y(lateral), ang_z(yaw)].
    # command_remap maps [-1,0,1] input onto [min,mid,max]. Kept within training's [-1,1] command
    # range; tune the max magnitudes to taste (they bound commanded speed, not the obs). This is the
    # ONE place vx/vy/wz range is clamped -- JoystickCtrl (sim), UnitreeCtrl (real) and KeyboardCtrl
    # all funnel through the same _update_velocity_command, so editing this (or a robot's own
    # commands_map override, e.g. G1UnifiedLocoKickPolicyCfg) retunes both sim and real identically.
    # A row's min/mid/max need not be increasing -- lin_y/ang_z below are deliberately reversed
    # (positive stick -> negative command) to match this project's axis-direction convention; the
    # validator below only requires each row be strictly monotonic, not increasing.
    commands_map: list[list[float]] = [
        [-0.8, 0.0, 0.8],  # forward/back  (LeftY / w,s)
        [0.5, 0.0, -0.5],  # left/right    (LeftX / a,d)
        [0.8, 0.0, -0.8],  # yaw           (RightX / q,e)
    ]

    @field_validator("commands_map")
    @classmethod
    def _check_commands_map(cls, v: list[list[float]]) -> list[list[float]]:
        axis_names = ["lin_x (fwd/back)", "lin_y (left/right)", "ang_z (yaw)"]
        if len(v) != 3:
            raise ValueError(f"commands_map must have exactly 3 rows [lin_x, lin_y, ang_z], got {len(v)}")
        for name, row in zip(axis_names, v):
            if len(row) != 3:
                raise ValueError(f"commands_map row '{name}' must be [min, mid, max], got {row}")
            lo, mid, hi = row
            increasing = lo < mid < hi
            decreasing = lo > mid > hi
            if not (increasing or decreasing):
                raise ValueError(
                    f"commands_map row '{name}' = {row} is not strictly monotonic -- "
                    "min/mid/max must be either increasing or decreasing"
                )
        return v

    # Seconds for the applied command to ramp from 0 to each axis's max magnitude when
    # ACCELERATING, rather than stepping there in a single tick. A raw instant step is
    # in-distribution enough to not destabilize the policy on its own, but side-by-side sim testing
    # against holosoma's own reference (whose keyboard scheme is a gradual +/-0.1-per-press
    # accumulator, never an instant jump) showed the gradual-ramp case tracking the smoothest and
    # closest to holosoma's trajectory of any command profile tried -- this reproduces that smoothing
    # for both keyboard and joystick without needing to replicate holosoma's specific accumulator UX.
    command_ramp_time: float = 0.5

    # DECELERATION is deliberately SLOWER than acceleration. Empirical result (v6 Stage-B
    # checkpoint, MuJoCo, keyboard stop-from-max-speed 0.8 m/s, 10 release phases spanning a full
    # gait cycle): instant cut fell at 8/10 phases, fast decel (0.15s) 8/10, the original
    # symmetric 0.5s 3/10, and 1.0s decel 0/10. Stopping from a fast walk is the hard transient
    # -- a longer decel keeps the robot passing through slower, fully in-distribution walking
    # speeds and enters standing from a much easier state. (An earlier version of this comment
    # argued the opposite -- that slow decel's "crawl regime" was out-of-distribution and fast
    # decel matched training's discrete command resampling. The phase-swept control experiment
    # disproved that: the crawl isn't the problem, the stop-from-speed is.) The snap band exists
    # only to cleanly cross the policy's zero_cmd_eps standing threshold at the end of the ramp;
    # keep it small so it doesn't recreate a discrete stop.
    command_decel_time: float = 1.0
    command_zero_snap: float = 0.02

    # --- decel-then-fire kick entry (2026-09-08, "Path A" in memory
    # loco_to_kick_handoff_sim2sim_drift_analysis_and_fix_strategy.md) ---
    # Training rehearses kick entry two ways -- a hard teleport at episode start, or Stage D's
    # state-matched search-then-blend mid-episode -- but deployment's own [TRIGGER_KICK] does
    # NEITHER: it keeps the robot's live walking state and snaps the reference straight to frame 0
    # with no state-matching and no blend, a third combination training never rehearses. That
    # mismatch is the leading suspected cause of a real, user-observed handoff hit-rate drop
    # (kicks reliably from a settled stand, misses more after a sudden mid-walk trigger).
    #
    # When True: [TRIGGER_KICK]/[TRIGGER_KICK:N] no longer fires immediately. Instead the
    # commanded velocity is forced to zero -- reusing the EXACT SAME, already-tuned
    # command_decel_time/command_zero_snap ramp above (empirically 0/10 falls stopping from 0.8
    # m/s over 1.0s, vs 8/10 for an instant cut -- see that field's own comment), not new ramp
    # arithmetic -- and the kick fires only once self.is_standing has held for kick_entry_settle_s
    # afterward. This converts entry into something close to mujoco_kick_survival_scan.py's own
    # settled-standstill start (reset -> 1.5s zero-vel settle -> trigger), which checkpoints that
    # actually kick already generalize to from training's teleported initial condition.
    #
    # Real costs, not free: (1) kick latency -- roughly command_decel_time + kick_entry_settle_s
    # from an initial fast walk, no longer instant; (2) Stage D's state-matched entry search goes
    # entirely unused at deployment -- sidestepping it, not using it, a real strategic choice; (3)
    # UNVALIDATED risk -- zero commanded velocity doesn't guarantee the same STANCE a keyframe
    # reset produces (a robot decelerating from a walking gait can settle mid-stride, asymmetric
    # weight/staggered feet), which frame 0 doesn't account for. Only a real Delta-hit_rate
    # measurement (paired kick_survival vs. loco_to_kick_handoff, the user's own stated success
    # criterion) resolves whether that risk matters in practice. Default False: zero behavior
    # change for every existing deployment until explicitly opted into and measured.
    kick_entry_decel_enabled: bool = False
    # Extra hold time AFTER the commanded velocity ramp reaches near-zero (self.is_standing),
    # before actually firing -- the ramp reaching zero-COMMAND doesn't mean the robot's real
    # physical velocity has finished settling yet. 0.5s is a starting point, not a measured
    # optimum; mujoco_kick_survival_scan.py's own reference settle window is longer (1.5s) but
    # that script settles from a hard keyframe reset, not a live decel -- tune against a real
    # Delta-hit_rate measurement rather than assuming either number transfers.
    kick_entry_settle_s: float = 0.5

    # --- "ball is in the SELECTED skill's trained range" readiness gesture ---
    # When ENABLED, and a live ball reading is available (--live-ball): while ball_pos_b's (x, y)
    # stays inside the CURRENTLY SELECTED skill's trained ball box -- skill_ball_xy[sel] +-
    # (randomize_x, randomize_y), read per-skill from the ONNX's experiment_config -- the right arm
    # swings CONTINUOUSLY, purely as a "the ball is where I expect it for this skill, I'm lined up"
    # signal for an operator. Only while in locomotion mode (never mid-kick) and -- if
    # ready_gesture_only_when_standing -- only while the commanded velocity is ~0. The swing eases
    # IN over ready_gesture_ramp_s when the ball enters the box and eases OUT over the same time
    # when it leaves, so it persists/repeats with no start/stop jerk. It superimposes a small
    # oscillation onto the right arm's pd_target ONLY; it does not change gains, gate the policy, or
    # affect balance/kick/anything else. No-op unless the checkpoint has skill_ball_xy metadata.
    ready_gesture_enabled: bool = False
    ready_gesture_ramp_s: float = 0.4  # ease-in / ease-out time for the swing amplitude
    ready_gesture_shoulder_amp_rad: float = 0.5  # right_shoulder_pitch swing amplitude
    ready_gesture_elbow_amp_rad: float = 0.6  # right_elbow swing amplitude, in phase with the shoulder
    ready_gesture_freq_hz: float = 1.2  # swings per second
    ready_gesture_only_when_standing: bool = True
    # per-skill box half-widths fall back to this (x, y) when the ONNX has no experiment_config to
    # read per-skill randomize_x/randomize_y from (0.1/0.1 is this project's standard default).
    ready_gesture_box_halfwidth_fallback_xy: list[float] = [0.1, 0.1]

    # --- one-shot "skill cycled" LEFT-arm wave ---
    # Every time [CYCLE_KICK_SKILL] (keyboard j / joystick RB+X) advances the pending skill
    # selection, the LEFT arm does a single brief swing and returns to neutral -- a visual
    # acknowledgment that the press registered (and a side cue: RIGHT arm = "ball is in range"
    # readiness gesture, LEFT arm = "skill cycled"). ONE-SHOT, not continuous: it plays
    # skill_cycle_gesture_duration_s of a windowed sine on the left shoulder/elbow pd_target and
    # then stops on its own. Locomotion-only (never overlaid on a running kick clip); pressing
    # cycle again while a wave is still playing restarts it from the top. Pure pd_target overlay --
    # it does NOT touch gains / self.last_action / balance / the kick. Independent of --live-ball
    # and of ready_gesture_enabled (they drive opposite arms and can both be on at once). No-op on
    # a single-skill checkpoint (nothing to cycle).
    skill_cycle_gesture_enabled: bool = False
    skill_cycle_gesture_duration_s: float = 0.6  # total length of the one-shot wave
    skill_cycle_gesture_shoulder_amp_rad: float = 0.55  # left_shoulder_pitch swing amplitude
    skill_cycle_gesture_elbow_amp_rad: float = 0.45  # left_elbow swing amplitude, in phase
    skill_cycle_gesture_swings: float = 1.0  # full sine periods within the window (1.0 = one there-and-back)

    # --- hip-roll drift correction (gentle pd_target pull-back during sustained strafe/turn) ---
    # Deployment-side mitigation for a real, measured checkpoint characteristic (2026-09-13
    # investigation, see UnifiedLocoKickPolicy's module docstring / _apply_hip_roll_correction's
    # own docstring for the full story): holding a CONTINUOUS non-zero lin_y/ang_z command for many
    # seconds lets hip_roll slowly drift and stance width grow past nominal (the training-time
    # standing-width guard is deliberately faded out any time a command is active, "so it never
    # fights stride-width variation while walking" -- correct for the gait, but nothing else bounds
    # the drift while it's faded). This is the SECOND mitigation attempt at this problem --
    # lateral_cooldown_* (above the autonav section) was the first, and an A/B MuJoCo comparison
    # showed it measurably WORSE, because reaching genuine zero necessarily crosses
    # _update_phase's is_standing boundary, retriggering a separate, already-documented gait-
    # phase-reset instability (see lateral_cooldown_enabled's own comment for the full story).
    # THIS approach avoids that failure mode BY CONSTRUCTION: it is a pure pd_target overlay on
    # hip_roll (same category as ready_gesture/skill_cycle_gesture) that never touches
    # lin_vel_command/ang_vel_command/_smoothed_cmd at all, so it cannot cross is_standing or
    # retrigger that lurch.
    #
    # VERIFIED IN SIM (2026-09-13, A/B MuJoCo, 7-skill checkpoint, 15s sustained strafe/yaw at
    # these default gain/deadband/max values): genuine improvement, not just "no worse" -- strafe's
    # end-of-hold hip_roll_R drift went from -6.3deg to -4.6deg, stance width end 0.292m to 0.285m,
    # yaw's hip_roll_R -4.1deg to -3.0deg, forward -3.7deg to -2.9deg -- with base_z_min UNCHANGED
    # (0.725m either way, no falls, no new destabilization introduced). Still UNVALIDATED ON REAL
    # HARDWARE -- sim2sim improvement is a strong signal, not a substitute for testing there, and
    # the gain/deadband/max values are a reasonable starting point from this one measurement, not a
    # tuned optimum. Default False: zero behavior change until explicitly opted into.
    hip_roll_correction_enabled: bool = False
    # radians of hip_roll deviation from this checkpoint's OWN natural resting value (a baseline
    # re-captured every tick self.is_standing is True -- NOT literal zero / default_dof_pos, since
    # this checkpoint's natural standing hip_roll is measurably nonzero, ~-2.6 to -3.2deg on the
    # right leg even at rest) beyond which the correction engages. Below this, normal gait/stride
    # variation is left alone.
    hip_roll_correction_deadband_rad: float = 0.05  # ~2.9 deg
    # proportional gain: pd_target offset (rad) subtracted per rad of deviation beyond the
    # deadband, before the max_rad clamp below.
    hip_roll_correction_gain: float = 0.5
    # hard cap on the corrective pd_target offset magnitude, regardless of how large the measured
    # deviation gets -- a safety ceiling, not the normal operating point.
    hip_roll_correction_max_rad: float = 0.15  # ~8.6 deg

    # --- manual kick_aim_theta override (operator dials aim from THIS process's own controller) ---
    # When ENABLED, kick_target_pos_b is computed INTERNALLY every tick as [kick_aim_theta /
    # kick_aim_theta_ref_deg, 0.0] from an operator-held angle the controller nudges -- instead of
    # taking it from the live ball-perception controller (BallPoseRedisCtrl/Ros2Ctrl/UdpCtrl)'s own
    # kick_target_pos_b reading, which is what happens when this is False (unchanged legacy
    # behavior). kick_ball_pos_b (the ball's own position cue) is untouched either way -- still live
    # if --live-ball is wired in, zero otherwise; this only ever overrides the AIM term. Mirrors
    # dummy_ball_perception.py's --kick-aim-enabled/--kick-aim-theta-deg/--kick-aim-theta-ref-deg,
    # just driven from THIS process's own keyboard/joystick instead of a second process, and
    # adjustable LIVE instead of fixed for the whole run. kick_aim_theta_ref_deg itself is NOT a cfg
    # field here -- it's read from the ONNX's own experiment_config (see __init__), same source of
    # truth the checkpoint was actually trained against, falling back to this project's stable 45.0
    # default only if that metadata is absent. No-op unless the loaded checkpoint's selected skill
    # was trained with SkillConfig.kick_aim_enabled=True -- getting that right is still on the
    # caller, same as the standalone script (nudging warns if it can't confirm this).
    # SIGN: positive kick_aim_theta = the robot's own LEFT (holosoma's atan2 convention -- see
    # UnifiedLocoKickPolicy's module docstring for the full source citation). The deploy keyboard/
    # joystick bindings deliberately map their LEFT/RIGHT key names to the correspondingly-signed
    # INC/DEC command, not to +/-, so a caller reading g1_unified_loco_kick_cfg.py's trigger dicts
    # directly should not assume INC means "more positive" reads as "more right."
    manual_kick_aim_enabled: bool = False
    manual_kick_aim_step_deg: float = 5.0  # degrees nudged per [KICK_AIM_THETA_INC]/[_DEC] press

    # --- auto-navigation: drive locomotion (vx, vy, yaw_rate) toward the kicking zone ---
    # When ENABLED and toggled on at runtime ([TOGGLE_AUTONAV], starts OFF), _update_velocity_command
    # computes its OWN velocity command every locomotion tick instead of reading w/a/s/d/stick input
    # -- a simple proportional loop closing the live ball_pos_b reading (--live-ball) onto the
    # CURRENTLY SELECTED skill's own trained ball box (the SAME box _selected_skill_ball_box()
    # already computes for the readiness gesture -- no new geometry, reuses that parsed metadata).
    # Commands ZERO (holds position) the instant ball_pos_b lands inside the box; it never triggers
    # the kick itself, that stays a manual [TRIGGER_KICK] -- see UnifiedLocoKickPolicy's module
    # docstring for the full control law and its two cancellation rules (manual input always wins;
    # a lost/stale ball reading freezes and cancels rather than extrapolating). Needs --live-ball;
    # no-op on a checkpoint without skill_ball_xy metadata for the selected skill.
    autonav_enabled: bool = False
    # Closing speed = min(autonav_kp_approach * gap, autonav_max_speed), where `gap` is the distance
    # from the ball to the box BOUNDARY (0 at the edge) -- NOT the distance to the box centre. This
    # is what stops the robot blowing through the small box: the commanded speed is already ~0 by
    # the time the ball reaches the zone. Lower kp_approach / max_speed for a gentler, slower
    # approach; raise them to close distance faster (at the cost of more overshoot risk).
    autonav_kp_approach: float = 1.5  # 1/s -- speed per metre of box gap
    autonav_max_speed: float = 0.35  # m/s -- cruise cap while still far from the box
    autonav_kp_yaw: float = 2.0  # yaw-rate gain (1/s); dropped entirely once within 0.15 m of the box
    # joystick stick-axis magnitude (LeftX/LeftY/RightX) below which input does NOT count as
    # "manual override" -- avoids false-cancelling auto-nav from stick center-noise/drift. Keyboard
    # has no equivalent (w/a/s/d/q/e are discrete press/release, never noisy).
    autonav_manual_deadzone: float = 0.05

    # --- lateral/yaw cooldown: periodic wait-for-zero-then-dwell during a long strafe/turn hold ---
    # Deployment-side mitigation for a real, measured checkpoint characteristic (2026-09-13
    # investigation, see UnifiedLocoKickPolicy's module docstring and _apply_lateral_cooldown's own
    # docstring for the full story): holding a CONTINUOUS non-zero lin_y and/or ang_z command for
    # many seconds lets hip_roll slowly drift and stance width grow past nominal -- confirmed in
    # MuJoCo sim2sim over a 15s hold, comparably across multiple checkpoints from this training
    # lineage (NOT specific to one checkpoint), with real hardware reportedly showing a more
    # visible version of the same tendency ("legs splitting") than the idealized sim contact model
    # does. Root cause is the training-time standing-width guard being DELIBERATELY faded out by
    # an exponential gate on total command magnitude any time a command is active, with nothing
    # else bounding the drift for as long as it stays faded. When ENABLED, this does not retrain or
    # patch that guard -- it just gives the policy's OWN already-trained recovery behavior a
    # repeated chance to re-engage, by forcing lin_y/ang_z's TARGET (NEVER lin_x -- forward alone
    # did not show this drift in the measurement above) toward zero once a hold runs long enough,
    # waiting for the ACTUAL applied command to genuinely reach near-zero (not just "commanded to
    # be zero" -- see lateral_cooldown_dwell_s's own comment for why a fixed-duration pulse was
    # tried first and measurably failed to do anything), holding there briefly, then resuming
    # manual input. Manual-input driving only; auto-nav is untouched.
    #
    # ⚠️ MEASURED WORSE, NOT BETTER, once actually reaching zero (2026-09-13, A/B MuJoCo, 7-skill
    # checkpoint, 15s sustained strafe/yaw): width/hip_roll drift and base-height dip were ALL
    # worse with this ON than OFF. Root cause: reaching genuine near-zero necessarily crosses
    # `_update_phase`'s `is_standing` boundary, and _compute_autonav_cmd's own docstring (this same
    # file) already documents that repeatedly crossing that boundary "force-resets the gait phase
    # mid-settle -- observed to cascade into the robot lurching... by tens of cm." Every cooldown
    # cycle deliberately manufactures exactly that transition (once going in, once coming back out)
    # -- this mitigation fixes a slow drift by repeatedly re-triggering a WORSE, already-documented
    # instability elsewhere in this same policy. Do not enable without addressing that interaction
    # first (e.g. a version that dwells WITHOUT crossing zero_cmd_eps, or an active pd_target
    # overlay that never touches the commanded velocity at all, unlike this one). Kept in the
    # codebase disabled, tested, and documented rather than deleted -- the underlying drift this
    # was built to fix is still real, just not fixed by this particular approach. Default False:
    # zero behavior change unless explicitly (and, given the above, inadvisably) enabled.
    lateral_cooldown_enabled: bool = False
    # |lin_y| or |ang_z| above this counts as "actively straffing/turning" for the hold-timer --
    # same spirit as autonav_manual_deadzone, keeps residual stick noise from ever starting a hold.
    lateral_cooldown_speed_threshold: float = 0.05
    # seconds of CONTINUOUS lin_y/ang_z above the threshold before a cooldown triggers. Resets to 0
    # the instant the operator's own command drops back below the threshold on its own -- this is
    # a ceiling on sustained-hold duration, not a fixed metronome independent of what's held.
    lateral_cooldown_hold_s: float = 5.0
    # once triggered, lin_y/ang_z's target is forced to 0 and held there UNTIL the actual applied
    # command (self._smoothed_cmd) is within zero_cmd_eps of zero -- that wait is NOT this field,
    # and takes as long as the existing command_decel_time ramp needs (a first version of this
    # feature used a fixed-duration pulse INSTEAD of waiting for real convergence: caught by an A/B
    # MuJoCo comparison showing byte-identical results with the mitigation on vs off, because the
    # 0.4s pulse was shorter than the 1.0s command_decel_time default -- the command never actually
    # reached zero before forcing stopped, so the standing guard never got a real chance to
    # re-engage). This field is the ADDITIONAL dwell time held at genuine near-zero once convergence
    # is confirmed, before resuming manual input -- the actual "let the guard work" window.
    lateral_cooldown_dwell_s: float = 0.5
