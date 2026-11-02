from robojudo.controller.ctrl_cfgs import CtrlCfg


class PingPongPerceptionCtrlCfg(CtrlCfg):
    """The player's input: perception's table-frame messages over UDP (see PingPongPerceptionCtrl)."""

    ctrl_type: str = "PingPongPerceptionCtrl"
    host: str = "0.0.0.0"
    port: int = 7780
    world_clock: bool = False
    """False: the robot (capture times on this computer's clock, from each message's age). True: a simulation (the world's clock)"""
    lockstep: bool = True  # simulation: wait each step until perception has answered the frames the world sent
    lockstep_timeout: float = 0.25  # s (wall)


class PingPongSimWorldCtrlCfg(CtrlCfg):
    """The simulated world: the opponent and the OptiTrack cameras (see PingPongSimWorldCtrl)."""

    ctrl_type: str = "PingPongSimWorldCtrl"
    send_to: str | None = "127.0.0.1:7770"  # the perception program's raw port; None: perception inside this process (batch runs)
    balls: str = "standard"  # a ball scenario (data/ball_scenarios/<name>) or a *_closed_loop_*.npz: its sequences in turn, then again
    env_idx: int = 0  # the scenario's sequence to start with (of 64, 8 balls each)
    random: str | None = None  # instead: fresh balls of this kind, drawn without end (scripts/planner/make_ball_scenarios.py: standard, topspin, ...)
    n_shots: int = 0  # 0: no end (stop with Ctrl-C or by closing the window)
    settle: float = 2.0  # s from the start of play to the first serve
    wait: float = 0.6  # s between a shot's end and the next serve (plus 0.6)
    sensing: str = "optitrack"  # optitrack (240 Hz) | optitrack120 (the lab's Motive stream, 2026-11-02) | camera | stereo
    opti_frame: str = "random"  # random: OptiTrack's frame is a random rotation + translation of the table frame; fixed: the same
    clutter: int = 2
    ball_stickers: int = 6  # the ball's dot stickers (as on the lab ball: OptiTrack sees 2-5 of them); 0: a ball covered in reflective
    # material (one marker at its centre). Perception's --ball must say the same (stickers / covered)
    pelvis_noise: float = 0.0005
    pelvis_pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)  # a Motive pivot off the pelvis origin (the pelvis origin seen from it, pelvis
    # axes; a test of the legs' height check and of run_perception.py --pelvis-pivot, which undoes it)
    table_markers: bool = True  # markers on the table's corners (for perception's calibration)
    table_rigid_body: bool = True  # ... tracked as Motive's "table" rigid body (the recommended real setup): sent apart from the unlabeled
    # markers, never taken for the ball, on during play (perception watches the table). False: plain markers, taken off when play starts
    table_layout: str = "corners+side_mids"  # where the table's markers sit: corners3 | corners4 | corners+edge_mids (4 corners on the top
    # + the middle of each edge, on the top) | corners+side_mids (4 corners on the top + the middle of each side, on the side faces: the
    # lab table, 8 markers)
    keep_table_markers: bool = False  # plain markers only: leave them on during play (the operator forgot them)
    seed: int = 0
