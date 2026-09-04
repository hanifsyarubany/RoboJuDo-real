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
    # range; tune the max magnitudes to taste (they bound commanded speed, not the obs).
    commands_map: list[list[float]] = [
        [-0.8, 0.0, 0.8],  # forward/back  (LeftY / w,s)
        [0.5, 0.0, -0.5],  # left/right    (LeftX / a,d)
        [0.8, 0.0, -0.8],  # yaw           (RightX / q,e)
    ]

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
