"""G1 environment configs that use the *holosoma-trained* MuJoCo model instead of RoboJuDo's stock
``g1_29dof_rev_1_0.xml``.

Why: the unified loco+kick policy was trained (IsaacSim) against holosoma's calibrated G1 model and
sim2sim-evaluated in holosoma's MuJoCo build of it. RoboJuDo's stock model is physically different
-- uniform placeholder armature 0.01 (vs holosoma's per-joint 0.0036-0.025 reflected rotor
inertias), MuJoCo-default contacts (vs holosoma's foot<->floor friction=0.8, solref=0.01), and
different torque limits (ankle 40 vs 50) -- so the *same* policy behaves differently. These configs
close that sim2sim gap by simulating the model the policy actually expects.

For real hardware (UnitreeCppEnv) the MuJoCo *dynamics* never run -- the real robot is the physics
-- but the model's *kinematics* are still used for forward kinematics (torso_quat ->
kick_motion_ref_ori_b). ``G1HolosomaRealFkCfg`` keeps that FK consistent with training too.
"""

from robojudo.config import ASSETS_DIR
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg

from .g1_env_cfg import G1_29DoF
from .g1_mujuco_env_cfg import G1MujocoEnvCfg
from .g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg

# Scene wrapper around holosoma's copied robot model (adds a ground plane the standalone robot XML
# lacks); the calibrated dynamics come from the included robot XML unchanged.
_HOLOSOMA_SCENE = (ASSETS_DIR / "robots/g1/holosoma_model/scene_g1_29dof.xml").as_posix()

# Torque limits = holosoma's actuator ctrlrange (Nm), in dof order. The env clips torque to these
# in Python; setting them equal to the XML's actuator ctrlrange keeps the two consistent (RoboJuDo's
# stock values over-clip the ankle at 40 vs holosoma's 50, which hurts standing balance).
_HOLOSOMA_TORQUE_LIMITS = [
    88, 88, 88, 139, 50, 50,  # left leg
    88, 88, 88, 139, 50, 50,  # right leg
    88, 50, 50,               # waist
    25, 25, 25, 25, 25, 5, 5,  # left arm
    25, 25, 25, 25, 25, 5, 5,  # right arm
]


class G1HolosomaDoF(G1_29DoF):
    # Same joint order/default/gains as stock G1 (gains are overridden by the policy from ONNX
    # metadata anyway); only the torque limits are corrected to holosoma's.
    torque_limits: list[float] | None = _HOLOSOMA_TORQUE_LIMITS


_HOLOSOMA_DOF = G1HolosomaDoF()
_HOLOSOMA_FK = ForwardKinematicCfg(
    xml_path=_HOLOSOMA_SCENE,
    debug_viz=False,
    kinematic_joint_names=_HOLOSOMA_DOF.joint_names,
)


class G1HolosomaMujocoEnvCfg(G1MujocoEnvCfg):
    """MuJoCo sim2sim on the holosoma-trained model, at holosoma's physics rate (2000 Hz)."""

    xml: str = _HOLOSOMA_SCENE
    dof: DoFConfig = _HOLOSOMA_DOF
    forward_kinematic: ForwardKinematicCfg | None = _HOLOSOMA_FK

    sim_dt: float = 0.0005  # 2000 Hz physics (matches holosoma run_sim's fps=2000)
    sim_decimation: int = 40  # -> 50 Hz control (== policy freq)
    update_with_fk: bool = True
    torso_name: str = "torso_link"


class G1HolosomaRealEnvCfg(G1RealEnvCfg):
    """Real G1 via unitree_cpp, but with FK driven by the holosoma model (kinematics consistent with
    training). Real dynamics are the physical robot; only FK (torso_quat) uses this model."""

    env_type: str = "UnitreeCppEnv"
    xml: str = _HOLOSOMA_SCENE
    dof: DoFConfig = _HOLOSOMA_DOF
    forward_kinematic: ForwardKinematicCfg | None = _HOLOSOMA_FK
    unitree: G1UnitreeCfg = G1UnitreeCfg(net_if="eth0")
    torso_name: str = "torso_link"
