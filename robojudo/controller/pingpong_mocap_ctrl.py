"""Table tennis: what the player receives (perception), and the simulated world that stands in for the real one.

Deployment is two programs (dual deployment). PERCEPTION (the pingpong folder's scripts/deploy/run_perception.py) turns OptiTrack's raw
frames into table-frame messages: the pelvis pose, the incoming ball's samples, shot start / end. The PLAYER (this RoboJuDo pipeline:
estimator, planner, swing clock, policy) only reads those messages. The same two programs run in simulation and on the robot:

    real        Motive -> natnet_relay.py --raw UDP 7770--> run_perception.py --UDP 7780--> PingPongPerceptionCtrl -> policy -> G1
    simulation  PingPongSimWorldCtrl (the MuJoCo world's cameras) --raw UDP 7770--> run_perception.py --UDP 7780--> ... -> MuJoCo robot

PingPongPerceptionCtrl   receives the perception messages (pingpong.deploy.perception). In a simulation it runs on the world clock and
                         in lockstep with the world (it waits, per step, until perception has answered the frames the world just sent),
                         so the simulation may run at any speed.
PingPongSimWorldCtrl     simulation only: the opponent (serves the scenario's balls into PingPongMujocoEnv) and the OptiTrack cameras (the
                         env's truth -> raw frames in a made-up OptiTrack frame, the table-corner markers on until play starts, so that
                         perception can calibrate). send_to=None keeps everything in one process instead (fast batch runs: g1_pingpong_inproc).
"""

import logging
import math
import sys
import time
from collections import deque

import numpy as np

from robojudo.config.g1.ctrl.g1_pingpong_ctrl_cfg import PingPongPerceptionCtrlCfg, PingPongSimWorldCtrlCfg
from robojudo.controller import Controller, ctrl_registry
from robojudo.utils.pingpong_path import ensure_pingpong

logger = logging.getLogger(__name__)
SENSING = {"optitrack": (240.0, 0.001, 0.010), "camera": (100.0, 0.005, 0.030), "stereo": (60.0, 0.010, 0.030)}


def world_time(env):
    """The simulated world's clock: the physics time plus the time spent waiting for perception before play started."""
    return float(env.data.time) + getattr(env, "t_offset", 0.0)


@ctrl_registry.register
class PingPongPerceptionCtrl(Controller):
    cfg_ctrl: PingPongPerceptionCtrlCfg

    def __init__(self, cfg_ctrl: PingPongPerceptionCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        ensure_pingpong()
        from pingpong.deploy.perception import PerceptionReceiver

        self.world = cfg_ctrl.world_clock
        self.clock = (lambda: world_time(self.env)) if self.world else time.perf_counter
        self.rx = PerceptionReceiver(port=cfg_ctrl.port, host=cfg_ctrl.host, clock=self.clock)
        self.max_src_seq, self.last_rx_wall, self.n_msgs, self.n_lockstep_timeouts = 0, None, 0, 0
        logger.info(f"[PingPongPerceptionCtrl] listening for perception on UDP {cfg_ctrl.host}:{self.rx.port}"
                    + (" (simulation: world clock, lockstep)" if self.world else ""))

    def _take(self):
        msgs = self.rx.poll()
        if msgs:
            self.last_rx_wall = time.perf_counter()
            self.n_msgs += len(msgs)
            self.max_src_seq = max(self.max_src_seq, max(m.src_seq for m in msgs))
        return msgs

    def get_data(self):
        msgs = self._take()
        # (lockstep only once play has started: before that, perception may not be running yet)
        target = getattr(self.env, "cam_last_seq", 0) if (self.world and self.cfg_ctrl.lockstep and getattr(self.env, "lockstep_on", False)) else 0
        if target > self.max_src_seq:  # simulation: wait until perception has answered every frame the world sent this step
            deadline = time.perf_counter() + self.cfg_ctrl.lockstep_timeout
            while self.max_src_seq < target and time.perf_counter() < deadline:
                time.sleep(0.0002)
                msgs += self._take()
            if self.max_src_seq < target:
                self.n_lockstep_timeouts += 1
        silent = np.inf if self.last_rx_wall is None else time.perf_counter() - self.last_rx_wall
        return {"msgs": msgs, "t": self.clock(), "silent_s": silent}

    def wait_for_pelvis(self, timeout, tick=None, every=5.0):
        """Block until perception sends a pelvis pose (or timeout s): returns that message or None. tick(): called meanwhile (a
        simulation's world keeps its cameras running)."""
        t0, last = time.perf_counter(), 0.0
        while time.perf_counter() - t0 < timeout:
            if tick is not None:
                tick()
            for m in self._take():
                if m.pelvis_pos is not None:
                    return m
            if time.perf_counter() - last > every:
                last = time.perf_counter()
                logger.warning(f"[PingPongPerceptionCtrl] waiting for perception on UDP {self.rx.port} "
                               f"({time.perf_counter() - t0:.0f} s; {self.n_msgs} messages so far, none with the pelvis)")
            time.sleep(1.0 / 240.0)
        return None

    def reset(self):
        pass


@ctrl_registry.register
class PingPongSimWorldCtrl(Controller):
    cfg_ctrl: PingPongSimWorldCtrlCfg

    def __init__(self, cfg_ctrl: PingPongSimWorldCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        ensure_pingpong()
        from pingpong import config
        from pingpong.deploy.emulator import MocapEmulator
        from pingpong.deploy.table_frame import TableFrame

        c = cfg_ctrl
        rng = np.random.default_rng(c.seed)
        hz, noise, latency = SENSING[c.sensing]
        if c.opti_frame == "random":
            yaw, origin = rng.uniform(-math.pi, math.pi), np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), 0.0])
        else:
            yaw, origin = 0.0, np.zeros(3)
        s, co = math.sin(yaw), math.cos(yaw)
        R = np.array([[co, -s, 0], [s, co, 0], [0, 0, 1.0]])
        self.emu = MocapEmulator(origin, R, hz=hz, noise=noise, latency=latency, pelvis_noise=c.pelvis_noise, clutter=c.clutter,
                                 rng=np.random.default_rng(c.seed + 552_113), ball_stickers=c.ball_stickers)
        nl, nr, fl = self.emu.table_markers()
        self.frame = TableFrame.from_markers(nl, nr, fl, near_edge_x=config.table()["table"]["near_edge_x"])  # for in-process runs only
        T = config.table()["table"]
        n0, L, hw, h = T["near_edge_x"], T["length"], T["width"] / 2, T["height"]
        corners = [(n0, hw, h), (n0, -hw, h), (n0 + L, hw, h), (n0 + L, -hw, h)]
        so, sd = 0.01, 0.015  # a marker on a side face: its centre 1 cm out from the face, 1.5 cm below the top
        layout = {"corners3": corners[:3], "corners4": corners,
                  "corners+edge_mids": corners + [(n0 + L / 2, hw, h), (n0 + L / 2, -hw, h), (n0, 0.0, h), (n0 + L, 0.0, h)],
                  "corners+side_mids": corners + [(n0 + L / 2, hw + so, h - sd), (n0 + L / 2, -hw - so, h - sd), (n0 - so, 0.0, h - sd),
                                                  (n0 + L + so, 0.0, h - sd)]}[c.table_layout]
        self.emu.table_pts = [self.emu.to_opti(q) for q in layout] if c.table_markers else None
        self.emu.table_rigid_body = c.table_rigid_body
        self.tx = None
        if c.send_to:
            from pingpong.deploy.mocap import MocapSender

            host, port = c.send_to.rsplit(":", 1)
            self.tx = MocapSender(host, int(port))
        env.t_offset, env.cam_last_seq, env.lockstep_on = 0.0, 0, False
        logger.info(f"[PingPongSimWorldCtrl] simulated OptiTrack: {c.sensing} {hz:g} Hz / {noise * 1000:g} mm / {latency * 1000:g} ms, frame "
                    f"{c.opti_frame} (table x axis at {math.degrees(yaw):+.0f} deg); "
                    + (f"raw frames -> perception at {c.send_to}" if self.tx else "perception in this process"))
        root = ensure_pingpong()
        sys.path.insert(0, str(root / "scripts" / "main"))
        import live_mujoco as lm

        self.P0, self.V0, self.W0, self.balls_desc, _ = lm.load_balls(type("A", (), {"balls": c.balls})())
        self.queue, self._cycle, self._hp = deque(), 0, None
        self.k, self.next_launch, self._was_active, self.playing = 0, None, False, False
        logger.info(f"[PingPongSimWorldCtrl] opponent: " + (f"fresh '{c.random}' balls" if c.random else f"'{c.balls}' ({self.balls_desc}), "
                    f"its {self.P0.shape[1]} sequences of {self.P0.shape[0]} in turn from {c.env_idx}")
                    + (f", {c.n_shots} shots" if c.n_shots > 0 else ", no end"))

    def _refill(self):
        """More balls for the opponent: the scenario's next sequence (wrapping around), or a fresh draw of `random` balls."""
        c = self.cfg_ctrl
        if c.random:
            if self._hp is None:
                import torch

                sys.path.insert(0, str(ensure_pingpong() / "scripts" / "planner"))
                import make_ball_scenarios as mbs
                from pingpong.planner.hitter_planner import HitterPlanner, PlannerCfg

                if c.random not in mbs.SCENARIOS:
                    raise ValueError(f"random: {c.random!r} is not a ball kind; have {sorted(mbs.SCENARIOS)}")
                self._mbs, self._hp, self._gen = mbs, HitterPlanner(PlannerCfg(), device="cpu"), torch.Generator().manual_seed(c.seed + 4242)
            p0, v0, w0 = self._mbs.draw(self._mbs.SCENARIOS[c.random], 64, self._gen, self._hp)
            legal = self._mbs.fly(self._hp, p0, v0, w0)[0]  # only balls that land on our half
            self.queue.extend((p0[i].numpy(), v0[i].numpy(), w0[i].numpy()) for i in range(p0.shape[0]) if bool(legal[i]))
        else:
            e = (c.env_idx + self._cycle) % self.P0.shape[1]
            self._cycle += 1
            self.queue.extend((self.P0[s_, e], self.V0[s_, e], self.W0[s_, e]) for s_ in range(self.P0.shape[0]))

    def _next_ball(self):
        while not self.queue:
            self._refill()
        return self.queue.popleft()

    def reset(self):
        pass

    @property
    def done(self):
        n = self.cfg_ctrl.n_shots
        return n > 0 and self.k >= n and self.env.shot is None

    def _cameras(self, t, hide_ball):
        from pingpong.deploy.library import quat_mat

        env = self.env
        self.emu.add_pelvis(t, env.data.qpos[:3].copy(), quat_mat(env.data.qpos[3:7]))
        frames = self.emu.poll(t, hide_ball=hide_ball)
        if self.tx is not None:
            for f in frames:  # world clock; the age is the cameras' latency
                env.cam_last_seq = self.tx.send(f.t, f.t_arrival - f.t, f.pelvis_pos, f.pelvis_quat, f.markers, world_clock=True,
                                                table_markers=f.table_markers)
        return frames

    def publish_waiting(self, t_world):
        """Before play: the world stands still (the robot in its ready pose, no ball) while its cameras run, so that perception can
        calibrate the table and start. t_world: the world clock (the launcher advances it with the wall clock)."""
        self.env.t_offset = t_world - float(self.env.data.time)
        self._cameras(t_world, hide_ball=True)

    def start_play(self):
        """Play begins (the launcher calls it): the operator has taken the table markers off; the first serve comes `settle` s later."""
        if not (self.cfg_ctrl.table_rigid_body or self.cfg_ctrl.keep_table_markers):  # plain markers come off; a table rigid body stays
            self.emu.table_pts = None
        self.playing = True
        self.next_launch = world_time(self.env) + self.cfg_ctrl.settle + self.cfg_ctrl.wait
        self.env.lockstep_on = self.tx is not None

    def get_data(self):
        env, c = self.env, self.cfg_ctrl
        t = world_time(env)
        if not self.playing:  # before play (the pipeline's start-up checks): the cameras only
            frames = self._cameras(t, hide_ball=True)
            return {"frames": [] if self.tx is not None else frames, "t": t, "table_frame": None if self.tx is not None else self.frame}
        active = env.shot is not None
        if self._was_active and not active:  # the last shot resolved: the opponent waits, then serves the next
            self.next_launch = t + c.wait + 0.6
        self._was_active = active
        if not active and (c.n_shots <= 0 or self.k < c.n_shots) and t >= self.next_launch - 1e-9:
            p0, v0, w0 = self._next_ball()
            env.launch_ball(self.k, np.asarray(p0, float), np.asarray(v0, float), np.asarray(w0, float))
            self.emu.add_ball(t, np.asarray(p0, float), np.asarray(w0, float))
            self.k += 1
            self._was_active = True
        off = getattr(env, "t_offset", 0.0)
        for ts, p, w in env.drain_ball_log():
            self.emu.add_ball(ts + off, p, w)
        frames = self._cameras(t, hide_ball=env.shot is None)
        if self.tx is not None:
            return {"frames": [], "t": t, "table_frame": None}
        return {"frames": frames, "t": t, "table_frame": self.frame}
