"""G1 environments for the table-tennis policy.

Sim: PingPongMujocoEnv on the ping-pong robot model (the paddle on the right wrist, the dummy left hand; assets/mjcf of the pingpong
package), with a SOLID table, the net and the analytic ball. The world frame is the table frame (home at the origin, the near table
edge 0.45 m ahead), so the born-place alignment is off. Real: the G1 over unitree_cpp, no odometry (the pelvis pose is OptiTrack's),
born-place alignment off (the IMU orientation and the encoders are used as they are).
"""

from typing import Literal

from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg
from robojudo.config.g1.env.g1_real_env_cfg import G1RealEnvCfg, G1UnitreeCfg
from robojudo.tools.tool_cfgs import ForwardKinematicCfg
from robojudo.utils.pingpong_path import ensure_pingpong

ensure_pingpong()
from pingpong import paths  # noqa: E402


class G1PingPongMujocoEnvCfg(G1MujocoEnvCfg):
    env_type: str = "PingPongMujocoEnv"
    xml: str = str(paths.MJCF_SCENE)
    forward_kinematic: ForwardKinematicCfg | None = None
    update_with_fk: bool = False
    visualize_extras: bool = False
    born_place_align: bool = False

    sim_dt: float = 0.0005  # 2000 Hz, as the pingpong evaluations
    sim_decimation: int = 40  # -> 50 Hz control
    headless: bool = True  # no viewer window
    solid_table: bool = True  # the table top, legs, net and posts collide with the robot


class G1PingPongUnitreeCfg(G1UnitreeCfg):
    enable_odometry: bool = False


class G1PingPongRealEnvCfg(G1RealEnvCfg):
    unitree: G1PingPongUnitreeCfg = G1PingPongUnitreeCfg(net_if="eth0")
    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "NONE"
    forward_kinematic: ForwardKinematicCfg | None = None
    update_with_fk: bool = False
    born_place_align: bool = False
