"""G1 table tennis: the PLAYER (estimator, planner, swing clock, policy). Perception is a separate program; see docs/g1_pingpong_deployment.md.

Dual deployment - the same two programs in simulation and on the robot:

    perception  (pingpong folder)  python scripts/deploy/run_perception.py run --calibrate-first
    player      (RoboJuDo)         python scripts/run_pingpong.py -c g1_pingpong        # simulation: MuJoCo, its cameras feed perception
                                   python scripts/run_pingpong.py -c g1_pingpong_real   # the robot: Motive's relay feeds perception

g1_pingpong         the player against MuJoCo. The simulated world (PingPongSimWorldCtrl) serves the balls and runs the OptiTrack cameras,
                    whose raw frames go to the perception program (UDP 7770); its messages come back (UDP 7780). World clock, lockstep.
g1_pingpong_real    the player on the G1 (unitree_cpp); perception's messages on UDP 7780. Stops: remote A = [SHUTDOWN], Select =
                    [SOFT_STOP], Start = [GUARD_STOP]. NET_IF below.
g1_pingpong_inproc  one process, no perception program (perception runs inside, on the world's raw frames): fast batch runs in sim.
"""

from robojudo.config import cfg_registry
from robojudo.config.g1.ctrl.g1_pingpong_ctrl_cfg import PingPongPerceptionCtrlCfg, PingPongSimWorldCtrlCfg
from robojudo.config.g1.env.g1_pingpong_env_cfg import G1PingPongMujocoEnvCfg, G1PingPongRealEnvCfg
from robojudo.config.g1.policy.g1_pingpong_policy_cfg import PingPongPolicyCfg
from robojudo.controller.ctrl_cfgs import UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

NET_IF = "eth0"  # the robot's network interface (real)


@cfg_registry.register
class g1_pingpong(RlPipelineCfg):
    """Simulation, run as the robot is: the world (MuJoCo + its cameras) -> the perception program -> this player."""

    robot: str = "g1"
    env: object = G1PingPongMujocoEnvCfg()
    ctrl: list = [PingPongSimWorldCtrlCfg(), PingPongPerceptionCtrlCfg(world_clock=True)]  # the world first: it sends before the player reads
    policy: PingPongPolicyCfg = PingPongPolicyCfg(threaded=True)
    do_safety_check: bool = False  # the launcher watches the pelvis height and tilt (scripts/run_pingpong.py)


def _real_env():
    cfg = G1PingPongRealEnvCfg()
    cfg.unitree.net_if = NET_IF
    return cfg


@cfg_registry.register
class g1_pingpong_real(RlPipelineCfg):
    """The real G1, fed by the perception program (UDP 7780)."""

    robot: str = "g1"
    env: object = _real_env()
    ctrl: list = [
        PingPongPerceptionCtrlCfg(),
        UnitreeCtrlCfg(triggers={"A": "[SHUTDOWN]", "Select": "[SOFT_STOP]", "Start": "[GUARD_STOP]"}),
    ]
    # noise: the lab's stickered ball scatters ~6 mm per axis (shot log 2026-11-02); with the 1 mm default the estimator takes the scatter
    # for spin (first plans 235-309 mm off); 7 mm: ~35 mm, sim with lab-like noise 89 % -> 94-97 % returned
    policy: PingPongPolicyCfg = PingPongPolicyCfg(threaded=True, noise=0.007)
    do_safety_check: bool = False  # the launcher watches the pelvis height and tilt (scripts/run_pingpong.py)


@cfg_registry.register
class g1_pingpong_inproc(RlPipelineCfg):
    """Simulation in one process (perception inside, planner jobs inline): for fast, repeatable batch runs."""

    robot: str = "g1"
    env: object = G1PingPongMujocoEnvCfg()
    ctrl: list = [PingPongSimWorldCtrlCfg(send_to=None)]
    policy: PingPongPolicyCfg = PingPongPolicyCfg()
    do_safety_check: bool = False
