"""G1 config for the holosoma unified locomotion + ball-kicking policy (UnifiedLocoKickPolicy).

The dof gains / action scales / joint order all come from the ONNX modelmeta at load time, so the
DoFConfig below is only a placeholder to satisfy PolicyCfg (the policy overrides it in __init__ ---
same pattern as G1BeyondMimicPolicyCfg). The joint order matches RoboJuDo's G1 env exactly.

``onnx_path`` and ``default_dof_pos`` are the only things you must get right per-checkpoint:
- onnx_path: the exported unified policy (a Stage-B checkpoint by default; a Stage-C checkpoint
  works too, and can now get a real dynamic ball reading via run_pipeline_prepared.py's --live-ball
  flag, see g1_unified_loco_kick_cfg.py's PERCEPTION section).
- default_dof_pos: the training robot config's default joint pose. dof_pos observations are
  measured relative to it, so it must match training exactly (verified against the golden obs).
"""

from robojudo.policy.policy_cfgs import UnifiedLocoKickPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

# Default: the Stage-B "ballobs-gated" checkpoint the user has been sim2sim-testing. Override with
# --policy.onnx-path <path> or by editing here.
# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kicking_skills/logs/"
#     "LocomotionAndBallKicking/20260801_150030-unified-stageC-2skills-locomotion/model_0437000.onnx"
# )
# DEFAULT_ONNX_PATH = (
#     "assets/motions/g1/football_play/"
#     "20260712_083233-unified-stageB-ballobs-gated-locomotion/model_0119000.onnx"
# )
# DEFAULT_ONNX_PATH = (
#     "assets/motions/g1/football_play/"
#     "20260711_032532-unified-stageA-locomotion-locomotion/model_0015000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "assets/motions/g1/football_play/"
#     "20260714_150605-unified-stageB-ballobs-gated-v10-locomotion/model_0145000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_robonaldo/logs/"
#     "RoboNaldoBallKicking/20260810_082502-stageD-1skill-locomotion/model_0330000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_robonaldo/logs/"
#     "RoboNaldoBallKicking/20260814_003043-stageD-1skill-handoff-locomotion/model_0385000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_robonaldo/logs/"
#     "RoboNaldoBallKicking/20260813_005001-stageC1-1skill-locoflip-shooting05-new-fixes-3-locomotion/model_0325000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_robonaldo/logs/"
#     "RoboNaldoBallKicking/20260814_021015-stageC-1skill-locoflip-shooting05-new-fixes-4-locomotion/model_0400000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/logs/"
#     "UnifiedBallKickingEnhanced/20260827_044728-stageB-skill011-h076-locomotion/model_0250000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "assets/motions/g1/football_play/"
#     "stageB-skill011/model_0250000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/logs/"
#     "UnifiedBallKickingEnhanced/20260827_021301-stageB-skill012-h074-locomotion/model_0200000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/logs/"
#     "UnifiedBallKickingEnhanced/20260827_044801-stageB-skill012-h074-locomotion/model_0250000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/logs/"
#     "UnifiedBallKickingEnhanced/20260903_042200-distill-4skills-12161718-distill/model_0250000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/logs/"
#     "UnifiedBallKickingEnhanced/20260904_021435-stageC-skill012-h074-locomotion/model_0430000.onnx"
# )

# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/logs/"
#     "UnifiedBallKickingEnhanced/20260901_115721-stageC-skill012-h074-locomotion/model_0390000.onnx"
# )

DEFAULT_ONNX_PATH = (
    "/workspaces/isaaclab_arena/submodules/workspaces/playground/unified_ball_kick_enhanced/logs/"
    "UnifiedBallKickingEnhanced/20260904_021435-stageC-skill012-h074-locomotion/model_0490000.onnx"
)




class G1UnifiedLocoKickDoF(DoFConfig):
    # 29-DoF G1, exactly the ONNX dof_names order (== RoboJuDo G1 env order). Placeholder only:
    # stiffness/damping/action_scale are read from the ONNX metadata by the policy.
    joint_names: list[str] = [
        *["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
          "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint"],
        *["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
          "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint"],
        *["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        *["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
          "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
        *["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
          "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
    ]

    # training robot config default pose (verified against golden default_dof_angles)
    default_pos: list[float] | None = [
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        *[-0.312, 0.0, 0.0, 0.669, -0.363, 0.0],
        *[0.0, 0.0, 0.0],
        *[0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
        *[0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
    ]


_DEFAULT_POSE = G1UnifiedLocoKickDoF().default_pos


class G1UnifiedLocoKickPolicyCfg(UnifiedLocoKickPolicyCfg):
    robot: str = "g1"

    onnx_path: str = DEFAULT_ONNX_PATH
    default_dof_pos: list[float] = _DEFAULT_POSE

    obs_dof: DoFConfig = G1UnifiedLocoKickDoF()
    action_dof: DoFConfig = G1UnifiedLocoKickDoF()

    # Set True to have the right arm swing CONTINUOUSLY while the live ball reading (--live-ball)
    # stays inside the currently-selected skill's trained ball box -- a "ball is where I expect it
    # for this skill, ready to kick" signal for the operator (eases in on entering the box, eases
    # out on leaving). No-op without --live-ball or on a checkpoint with no skill_ball_xy metadata.
    # See UnifiedLocoKickPolicyCfg for the full knob set (ramp / amplitude / frequency /
    # only-when-standing).
    ready_gesture_enabled: bool = True

    # Set True to have the LEFT arm do a single brief wave every time [CYCLE_KICK_SKILL] (keyboard
    # j / joystick RB+X) advances the pending skill selection -- a one-shot "the cycle press
    # registered" acknowledgment (and a side cue vs. the right-arm readiness gesture above). No
    # --live-ball needed; no-op on a single-skill checkpoint. See UnifiedLocoKickPolicyCfg for the
    # duration / amplitude / swing-count knobs.
    skill_cycle_gesture_enabled: bool = True

    # Set True to let keyboard `,`/`.`/`0` (joystick LB+Left/LB+Right/LB+Down) dial in kick_aim_theta
    # live from THIS process's own controller -- no second dummy_ball_perception.py process or
    # restart needed to try a different aim angle. Overrides kick_target_pos_b only; kick_ball_pos_b
    # still comes from --live-ball (or zero) unaffected. See UnifiedLocoKickPolicyCfg for the
    # step-size knob and the per-skill clamp/fallback behavior.
    manual_kick_aim_enabled: bool = True

    # Set True to let 'n' (joystick LB+Up) toggle auto-navigation: while ON, the policy drives its
    # own (vx, vy, yaw_rate) to walk the robot into the currently-selected skill's trained ball box,
    # holds once arrived, and leaves triggering the kick to the operator. Needs --live-ball; any
    # manual stick/keyboard input instantly cancels it. See UnifiedLocoKickPolicyCfg for the gain
    # knobs (autonav_kp_x/y/yaw) and UnifiedLocoKickPolicy's module docstring for the full design.
    autonav_enabled: bool = True
