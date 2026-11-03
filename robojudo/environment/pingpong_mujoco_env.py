"""MujocoEnv for table tennis: the ping-pong robot model with a solid table, the analytic ball (pingpong.sim2sim.mj_ball) stepped at
200 Hz inside every control step, and the shot's outcome. The robot, its PD law, the guard / estop machinery are MujocoEnv's.

The ball: launch_ball() serves one; each control step the ball advances 4 x 5 ms against the paddle face and the body capsules (poses
taken before / after each 5 ms of physics), exactly as scripts/main/live_mujoco.py does. `shot` is the ball in play (None between
shots); when it resolves (returned over the net, into the net, missed, ...) its outcome goes to `results`. `ball_log` holds the
ball's true positions for the mocap emulator (drain_ball_log()).
"""

import logging
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from robojudo.environment import Environment, env_registry
from robojudo.environment.mujoco_env import MujocoEnv
from robojudo.utils.pingpong_path import ensure_pingpong

logger = logging.getLogger(__name__)
FALL_Z = 0.45  # the pelvis below this: fallen (eval_mujoco.FALL_Z)


class _NoViewer:
    """MujocoViewer stand-in for runs without a window."""

    is_alive = False

    def __init__(self, *a, **k):
        self.cam = SimpleNamespace(distance=0.0, elevation=0.0, azimuth=0.0, lookat=np.zeros(3))

    def render(self):
        pass

    def close(self):
        pass


@env_registry.register
class PingPongMujocoEnv(MujocoEnv):
    def __init__(self, cfg_env, device="cpu"):
        ensure_pingpong()
        from pingpong.sim2sim.mj_ball import BALL_DT, MjBall

        Environment.__init__(self, cfg_env=cfg_env, device=device)  # MujocoEnv.__init__ builds a window and a fixed model: done here
        self.sim_duration, self.sim_dt, self.sim_decimation = cfg_env.sim_duration, cfg_env.sim_dt, cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation
        spec = mujoco.MjSpec.from_file(cfg_env.xml)
        if cfg_env.solid_table:
            for g in spec.geoms:
                if g.name.startswith("table_"):
                    g.contype, g.conaffinity = 1, 1
        if cfg_env.paddle_extra_mass:  # a heavier paddle than the model's (the real one was never weighed): a point mass at the blade centre
            for b in spec.bodies:
                if b.name == "right_paddle_face":
                    b.mass = b.mass + cfg_env.paddle_extra_mass
        self.model = spec.compile()
        self.model.opt.timestep = self.sim_dt
        if cfg_env.foot_floor_friction is not None:
            self._override_foot_floor_friction(cfg_env.foot_floor_friction)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_step(self.model, self.data)
        if cfg_env.headless:
            self.viewer = _NoViewer()
        else:
            import mujoco_viewer

            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data, width=1200, height=900, hide_menus=True, diable_key_callbacks=True)
            # behind the robot, looking down the table (scripts/main/live_mujoco.py's viewer); drag with the mouse to move it
            self.viewer.cam.azimuth, self.viewer.cam.elevation, self.viewer.cam.distance = 0.0, -14.0, 2.8
            self.viewer.cam.lookat[:] = (0.6, 0.0, 0.7)
            # no vsync: with it every frame waits for the display's refresh, and a remote or hidden display may refresh once a second
            # (measured: 1000 ms per buffer swap with vsync, 0.3 ms without) - the simulation must never wait for the screen
            import glfw

            glfw.make_context_current(self.viewer.window)
            glfw.swap_interval(0)
        self.visualizer = None
        self.last_time = time.time()
        self.random_heading = False
        self._pre_estop_gains = None
        self._estop_ramp = None
        self._guard_ramp = None
        # the ball
        self.mb = MjBall(self.model)
        self.ball = self.mb.sim
        self.ball.active[:] = False
        self.ball_every = int(round(BALL_DT / self.sim_dt))
        self.ball_dt = BALL_DT
        self.shot = None
        self.results = []
        self.ball_log = []
        from pingpong import config

        self.ball_radius = config.ball()["ball"]["diameter"] / 2
        self.viz_fn = None  # () -> [marker dicts] drawn on top (the launcher: the racket target, the station)
        self.update()

    # ---- the robot ---------------------------------------------------------------------------------------------------------
    def reset_to_ready(self, root_z, root_quat_wxyz, q_motor, xy=(0.0, 0.0), yaw_deg=0.0):
        """Stand the robot at home in the swing library's ready pose (q_motor: the joint angles in the env's joint order)."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (xy[0], xy[1], root_z)
        h = np.radians(yaw_deg) / 2.0
        w0, x0, y0, z0 = root_quat_wxyz
        self.data.qpos[3:7] = (np.cos(h) * w0 - np.sin(h) * z0, np.cos(h) * x0 - np.sin(h) * y0,
                               np.cos(h) * y0 + np.sin(h) * x0, np.cos(h) * z0 + np.sin(h) * w0)
        self.data.qpos[7:7 + self.num_dofs] = q_motor
        mujoco.mj_forward(self.model, self.data)
        self.update()

    @property
    def fallen(self):
        return float(self.data.qpos[2]) < FALL_Z

    def shutdown(self):
        self.viewer.close()

    def _draw_markers(self):
        """The ball (orange; it is the analytic BallSim, not a MuJoCo body, so it is drawn), home (grey) and the launcher's markers."""
        v, S = self.viewer, mujoco.mjtGeom
        r = self.ball_radius
        pos = self.ball.pos[0].numpy().astype(float) if (self.shot is not None and bool(self.ball.active[0])) else np.array([0.0, 0.0, -5.0])
        v.add_marker(id=0, type=S.mjGEOM_SPHERE, size=[r, r, r], pos=pos, rgba=[1.0, 0.45, 0.0, 1.0], label="")
        v.add_marker(id=1, type=S.mjGEOM_CYLINDER, size=[0.12, 0.12, 0.001], pos=[0.0, 0.0, 0.002], rgba=[0.6, 0.6, 0.6, 0.4], label="")
        for k, m in enumerate(self.viz_fn() if self.viz_fn else []):
            v.add_marker(id=2 + k, label="", **m)

    # ---- the ball ----------------------------------------------------------------------------------------------------------
    def launch_ball(self, idx, p0, v0, w0):
        f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)[None]  # noqa: E731
        self.ball.reset(torch.tensor([0]), f(p0), f(v0), f(w0))
        self.shot = dict(idx=idx, t_launch=float(self.data.time), hit=False, p0=np.asarray(p0, float))

    def drain_ball_log(self):
        out, self.ball_log = self.ball_log, []
        return out

    def _ball_events(self, ev_any):
        """The outcome rules of eval_mujoco.py / live_mujoco.py, once per control step."""
        s = self.shot
        t = float(self.data.time)
        ph = ev_any.get("paddle_hit", False)
        after_hit = s["hit"] and not ph
        far = after_hit and ev_any.get("table_hit", False) and ev_any.get("side", 0) > 0
        bad = after_hit and (ev_any.get("net_hit", False) or ev_any.get("floor_hit", False)
                             or (ev_any.get("table_hit", False) and ev_any.get("side", 0) < 0))
        miss = (not s["hit"] and not ph) and (ev_any.get("floor_hit", False) or float(self.ball.pos[0, 0]) < -0.6 or (t - s["t_launch"]) > 2.5)
        timeout = s["hit"] and (t - s["t_launch"]) > 3.5
        s["hit"] = s["hit"] or ph
        if far or bad or miss or timeout:
            self.results.append(dict(idx=s["idx"], t_launch=s["t_launch"], t_end=t, contact=s["hit"], returned=bool(far),
                                     net=bool(after_hit and ev_any.get("net_hit", False)), body=bool(ev_any.get("body_hit", False))))
            self.shot = None
            self.ball.active[:] = False

    # ---- one control step --------------------------------------------------------------------------------------------------
    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
        if self.viewer.is_alive:
            self._draw_markers()
            self.viewer.render()
        self._apply_estop_ramp()
        pd_target = self._apply_guard_ramp(pd_target)
        before = self.mb.poses(self.data)
        ev_any = {}
        for i in range(self.sim_decimation):
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            self.data.ctrl = np.clip(torque, -self.torque_limits, self.torque_limits)
            mujoco.mj_step(self.model, self.data)
            self.update(simple=True)
            if (i + 1) % self.ball_every == 0:
                after = self.mb.poses(self.data)
                ev = self.mb.step(self.ball_dt, before, after)
                before = after
                for k in ("paddle_hit", "table_hit", "net_hit", "floor_hit", "body_hit"):
                    ev_any[k] = ev_any.get(k, False) or bool(ev[k][0])
                if bool(ev["table_hit"][0]):
                    ev_any["side"] = float(ev["table_hit_side"][0])
                if self.shot is not None:
                    self.ball_log.append((float(self.data.time), self.ball.pos[0].numpy().astype(float).copy(),
                                          self.ball.spin[0].numpy().astype(float).copy()))  # (t, position, spin)
        self.update(simple=False)
        if self.shot is not None:
            self._ball_events(ev_any)
