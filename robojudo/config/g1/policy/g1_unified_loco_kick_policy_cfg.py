"""G1 config for the holosoma unified locomotion + ball-kicking policy (UnifiedLocoKickPolicy).

The dof gains / action scales / joint order all come from the ONNX modelmeta at load time, so the
DoFConfig below is only a placeholder to satisfy PolicyCfg (the policy overrides it in __init__ ---
same pattern as G1BeyondMimicPolicyCfg). The joint order matches RoboJuDo's G1 env exactly.

``onnx_path`` and ``default_dof_pos`` are the only things you must get right per-checkpoint:
- onnx_path: the exported unified policy (a Stage-B checkpoint by default; a Stage-C checkpoint
  works too, but a Stage-C deploy would additionally need a real dynamic ball reading -- not wired
  here yet, see the holosoma repo README v9).
- default_dof_pos: the training robot config's default joint pose. dof_pos observations are
  measured relative to it, so it must match training exactly (verified against the golden obs).
"""

from robojudo.policy.policy_cfgs import UnifiedLocoKickPolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

# Default: the Stage-B "ballobs-gated" checkpoint the user has been sim2sim-testing. Override with
# --policy.onnx-path <path> or by editing here.
# DEFAULT_ONNX_PATH = (
#     "/workspaces/isaaclab_arena/submodules/workspaces/playground/locomotion_and_ball_kicking/logs/"
#     "LocomotionAndBallKicking/20260712_083233-unified-stageB-ballobs-gated-locomotion/model_0119000.onnx"
# )
# DEFAULT_ONNX_PATH = (
#     "assets/motions/g1/football_play/"
#     "20260712_083233-unified-stageB-ballobs-gated-locomotion/model_0119000.onnx"
# )
# DEFAULT_ONNX_PATH = (
#     "assets/motions/g1/football_play/"
#     "20260711_032532-unified-stageA-locomotion-locomotion/model_0015000.onnx"
# )

DEFAULT_ONNX_PATH = (
    "assets/motions/g1/football_play/"
    "20260714_150605-unified-stageB-ballobs-gated-v10-locomotion/model_0145000.onnx"
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
