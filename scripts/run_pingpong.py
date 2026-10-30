"""The table-tennis PLAYER (estimator, planner, swing clock, policy) on MuJoCo or the real G1. Perception is its own program.

Dual deployment: two programs, the same in simulation and on the robot (see docs/g1_pingpong_deployment.md):

    1. perception (pingpong folder):  python scripts/deploy/run_perception.py run --calibrate-first
    2. this player (RoboJuDo):        python scripts/run_pingpong.py -c g1_pingpong [--window --realtime]   # simulation: serves without end
                                      python scripts/run_pingpong.py -c g1_pingpong_real [--dry-run]        # the robot

Either may start first. The player waits (--wait-perception s) for perception to report the pelvis before anything moves: in simulation the
world stands still meanwhile (the robot in its ready pose, no ball) and its cameras run, so that perception can calibrate the table; on
the robot nothing is commanded until then. -c g1_pingpong_inproc: one process, no perception program (fast batch runs in simulation).

Real robot: start it hanging from the gantry (feet just off the floor), then lower it once the policy holds the ready pose. prepare() ramps
the joints from wherever they are to the swing library's READY pose (not the default standing pose: the policy was always started there)
in --prepare-seconds, then blends the policy in over the same time; the policy holds the ready pose while no ball comes.

Stops. On the robot every stop is RoboJuDo's guard stop (rl_pipeline.stop_with_guard_pose: legs and waist to zero stiffness with
damping, at once; arms to the guard pose in 0.4 s). The guard is carried out by the control loop each tick and never clears by
itself, so after a stop this program keeps the loop running, holding it, until Ctrl-C; then it ends and the motors are released
(env.shutdown). What stops it:
    the remote: A, Select or Start (triggers in g1_pingpong_real) - during play and during the start-up ramp and blend (read from
                the robot's raw remote state there: RoboJuDo's prepare() does not read the remote in its ramp)
    Ctrl-C      in the terminal (a second Ctrl-C then ends the program)
    watchdogs   no fresh pelvis pose from perception for --stale-steps control steps; pelvis below 0.45 m or tilted more than
                --max-tilt-deg; a NaN in the targets
A stop needs this program running: if it is killed or hangs, the last command stays. Unitree's own stop (power) is the last resort.

Simulation viewer (--window): the ball orange, the racket target blue (during a swing), the station green, home grey.
"""

import os

# one math thread per process (set before numpy loads): the actor's matrix products, the planner and the simulator otherwise fight over
# the cores through a multi-threaded BLAS, and a 0.1 ms product takes 8 ms now and then
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys as _sys

if "--window" in _sys.argv and os.environ.get("MUJOCO_GL") in ("egl", "osmesa"):
    os.environ.pop("MUJOCO_GL")  # an off-screen GL backend and a window do not mix: the viewer opens its own (GLFW) context

import argparse
import json
import logging
import sys
import time

import struct

import numpy as np

import robojudo.pipeline  # noqa: F401  (torch before mujoco, see run_pipeline_prepared.py)
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


class ActionMonitor:
    """The dry run's eyes. Warns (once per kind) on a NaN, a torque the PD law would ask beyond the motor's limit, or a big jump between
    two steps; counts, per joint, how often and by how much the target lies outside the joint range (the policy's PD targets do, in
    training too: the joint sits on its stop and the motor pushes into it with kp x the excess)."""

    def __init__(self, env, policy):
        self.env, self.policy, self.prev, self.n = env, policy, None, 0
        self.torque_over = np.zeros(env.num_dofs)
        self.out_n = np.zeros(env.num_dofs)
        self.out_max = np.zeros(env.num_dofs)
        self.warned = set()

    def _warn(self, kind, msg):
        if kind not in self.warned:
            self.warned.add(kind)
            logger.warning(f"[ActionMonitor] step {self.n}: {msg} (further ones counted, not printed)")

    def check(self, target_env, q, dq):
        ok = True
        if not np.all(np.isfinite(target_env)):
            self._warn("nan", "NaN / inf in the joint targets")
            ok = False
        pl = getattr(self.env, "position_limits", None)
        if pl is not None and np.ndim(pl) == 2:
            exc = np.maximum(pl[:, 0] - target_env, 0.0) + np.maximum(target_env - pl[:, 1], 0.0)
            self.out_n += exc > 1e-3
            self.out_max = np.maximum(self.out_max, exc)
        tau = (target_env - q) * self.env.stiffness - dq * self.env.damping
        over = np.abs(tau) > self.env.torque_limits * 1.001
        self.torque_over += over
        if over.sum() > 4:
            self._warn("torque", f"{int(over.sum())} joints would exceed their torque limit")
        if self.prev is not None and np.max(np.abs(target_env - self.prev)) > 1.0:
            self._warn("jump", f"target jump {np.max(np.abs(target_env - self.prev)):.2f} rad in one step")
        self.prev = np.array(target_env)
        self.n += 1
        return ok

    def report(self):
        rows = [(self.env.joint_names[i], self.out_n[i] / max(self.n, 1), self.out_max[i], self.out_max[i] * self.env.stiffness[i], self.env.torque_limits[i])
                for i in np.argsort(-self.out_max) if self.out_n[i] > 0]
        if not rows:
            return
        print("\njoint targets outside the joint range (share of steps, largest excess, motor pushing into the stop at that excess = kp x excess):")
        for name, share, mx, tq, lim in rows[:8]:
            print(f"  {name:26s} {share:6.1%}   {mx:.3f} rad   {tq:5.1f} N m (limit {lim:.0f})")


class StageTimer:
    """Wall time of the stages of one control step (ms): where the 20 ms go. Wraps methods of the pipeline's parts."""

    def __init__(self):
        self.t = {}

    def wrap(self, obj, name, label):
        f = getattr(obj, name)

        def g(*a, **k):
            t0 = time.perf_counter()
            r = f(*a, **k)
            self.t.setdefault(label, []).append((time.perf_counter() - t0) * 1000.0)
            return r

        setattr(obj, name, g)

    def report(self, budget_ms):
        print(f"\nwall time per control step [ms] (budget {budget_ms:.0f}):   p50    p99    max")
        for k, v in self.t.items():
            v = np.array(v)
            print(f"  {k:34s} {np.percentile(v, 50):6.2f} {np.percentile(v, 99):6.2f} {v.max():6.2f}")


def tilt_deg(base_quat_xyzw):
    from robojudo.utils.util_func import get_gravity_orientation

    g = get_gravity_orientation(base_quat_xyzw)
    return float(np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0))))


STOP_COMMANDS = ("[SHUTDOWN]", "[SOFT_STOP]", "[GUARD_STOP]", "[ESTOP]")
REMOTE_BITS = {b: i for i, b in enumerate(["R1", "L1", "Start", "Select", "R2", "L2", "F1", "F2", "A", "B", "X", "Y", "Up", "Right",
                                           "Down", "Left"])}  # the remote's button word (robojudo/controller/utils/joystick.py)


class StopRequested(Exception):
    pass


def remote_stop_buttons(env, triggers):
    """The stop buttons held right now, read from the robot's raw remote state (not through the controller's event queue, which the
    pipeline drains itself): [] if none or if the state cannot be read."""
    try:
        keys = struct.unpack("H", bytes(env.unitree.get_robot_state().wireless_remote[2:4]))[0]
    except Exception:
        return []
    return [b for b, c in triggers.items() if c in STOP_COMMANDS and b in REMOTE_BITS and (keys >> REMOTE_BITS[b]) & 1]


def prepare_watching_stops(pipeline, env, triggers, prepare_seconds):
    """pipeline.prepare() (ramp to the ready pose, blend the policy in) with the remote's stop buttons watched every tick: a stop
    button held for 2 ticks (one corrupted radio packet does not stop) raises StopRequested before the next command is sent."""
    orig, held, own = env.step, [0], "step" in vars(env)

    def step(*args, **kw):
        b = remote_stop_buttons(env, triggers)
        held[0] = held[0] + 1 if b else 0
        if held[0] >= 2:
            raise StopRequested(f"operator stop {b} during start-up")
        return orig(*args, **kw)

    env.step = step
    try:
        pipeline.prepare(prepare_seconds=prepare_seconds)
    finally:  # the env's own step back
        if own:
            env.step = orig
        else:
            del env.step


def hold_guard(pipeline, env, dt, reason):
    """After a stop on the robot: arm RoboJuDo's guard stop (unless a stop command already did) and keep stepping the env, which carries
    it out (legs limp and damped, arms to the guard pose and held), until Ctrl-C."""
    if getattr(env, "_guard_ramp", None) is None:
        pipeline.stop_with_guard_pose(reason)
    logger.error(f"STOPPED ({reason}): legs limp and damped, arms in the guard pose, held. Ctrl-C to end the program "
                 "(the motors are then released).")
    try:
        while True:
            t0 = time.perf_counter()
            env.update()
            env.step(np.asarray(env.dof_pos, dtype=np.float32))
            time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        logger.warning("ended after the stop")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default="g1_pingpong")
    ap.add_argument("--dry-run", action="store_true", help="compute observations and actions every step, apply no torque")
    ap.add_argument("--prepare-seconds", type=float, default=3.0, help="real: the ramp to the ready pose, and again for the blend-in")
    ap.add_argument("--stale-steps", type=int, default=15, help="stop after this many control steps (20 ms) without a fresh pelvis pose")
    ap.add_argument("--max-tilt-deg", type=float, default=40.0)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="stop after this long (sim time in sim); 0: no limit")
    ap.add_argument("--window", action="store_true", help="sim: open the MuJoCo viewer")
    ap.add_argument("--realtime", action="store_true", help="sim: pace to real time (default: as fast as possible)")
    ap.add_argument("--log", type=str, default=None, help="write the run's per-step record and shot results here (.npz / .json)")
    ap.add_argument("--wait-perception", type=float, default=300.0, help="s to wait for the perception program before giving up")
    ap.add_argument("--perception-port", type=int, default=None, help="UDP port perception's messages come to (7780)")
    ap.add_argument("--raw-to", type=str, default=None, help="sim: host:port the simulated cameras send raw frames to (127.0.0.1:7770)")
    # the simulated world's options (PingPongSimWorldCtrlCfg)
    ap.add_argument("--balls", type=str, default=None)
    ap.add_argument("--env-idx", type=int, default=None)
    ap.add_argument("--n-shots", type=int, default=None, help="sim: stop after this many balls (default: no end; Ctrl-C or close the window)")
    ap.add_argument("--random", type=str, default=None, help="sim: fresh balls of this kind (standard, topspin, short, ...) instead of a scenario file")
    ap.add_argument("--sensing", choices=["optitrack", "camera", "stereo"], default=None)
    ap.add_argument("--opti-frame", choices=["random", "fixed"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--compute-scale", type=float, default=None, help="x the measured planner time (0 = instant)")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--actor-path", type=str, default=None)
    ap.add_argument("--start-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("FWD", "LEFT", "YAW_DEG"), help="sim: start off home")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = ConfigManager(config_name=a.config).get_cfg()
    sim = bool(cfg.env.is_sim)
    for c in cfg.ctrl:
        if c.ctrl_type == "PingPongSimWorldCtrl":
            for k in ("balls", "env_idx", "n_shots", "random", "sensing", "opti_frame", "seed"):
                if getattr(a, k, None) is not None:
                    setattr(c, k, getattr(a, k))
            if a.raw_to and c.send_to:
                c.send_to = a.raw_to
        if c.ctrl_type == "PingPongPerceptionCtrl" and a.perception_port:
            c.port = a.perception_port
    if a.compute_scale is not None:
        cfg.policy.compute_scale = a.compute_scale
    if a.checkpoint:
        cfg.policy.checkpoint = a.checkpoint
    if a.actor_path:
        cfg.policy.actor_path = a.actor_path
    if sim:
        cfg.env.headless = not a.window
    if not sim and not a.dry_run:
        logger.warning("REAL ROBOT, torque will be applied. Gantry rope on? Remote in hand (A / Select / Start = stop: legs limp, arms "
                       "guard pose, held until Ctrl-C)?")

    pipeline = RlPipeline(cfg)
    env, policy = pipeline.env, pipeline._inner_policy()
    ctrls = {c.inst.__class__.__name__: c.inst for c in pipeline.ctrl_manager.controllers.values()}
    sim_ctrl, perc = ctrls.get("PingPongSimWorldCtrl"), ctrls.get("PingPongPerceptionCtrl")
    lib, jmap = policy.rt.lib, policy.rt.jmap
    if sim:
        env.reset_to_ready(float(lib.ready_root_z[0]) + 0.016, lib.ready_root_quat[0], jmap.to_motor(lib.ready_q[0]),
                           xy=a.start_offset[:2], yaw_deg=a.start_offset[2])
    if perc is not None:  # dual deployment: nothing moves before perception reports the pelvis
        print("\n=== waiting for the perception program (it may start after this one). In the pingpong folder:\n"
              "    python scripts/deploy/run_perception.py run --calibrate-first" + ("" if sim else "      (or: run, with a calibration file)")
              + f"\n    (it sends to UDP {perc.rx.port}; {'the simulated world sends it raw frames on 127.0.0.1:7770' if sim else 'Motive relay -> UDP 7770'})\n",
              flush=True)
        tick = None
        if sim and sim_ctrl is not None:  # the world stands still, its cameras run (perception calibrates on them)
            t_wait0 = time.perf_counter()

            last_render = [0.0]

            def tick():
                sim_ctrl.publish_waiting(time.perf_counter() - t_wait0)
                if env.viewer.is_alive and time.perf_counter() - last_render[0] > 1 / 30:  # the window at 30 Hz while waiting
                    last_render[0] = time.perf_counter()
                    env._draw_markers()
                    env.viewer.render()

        m = perc.wait_for_pelvis(a.wait_perception, tick=tick)
        if m is None:
            logger.error(f"no pelvis pose from perception within {a.wait_perception:.0f} s: is run_perception.py running and sending to "
                         f"UDP {perc.rx.port}? Nothing was moved.")
            env.shutdown()
            return 1
        src = perc.rx.last_addr
        msg = (f"perception is up ({'messages from ' + src[0] if src else ''}): pelvis at ({m.pelvis_pos[0]:+.3f}, {m.pelvis_pos[1]:+.3f}, "
               f"{m.pelvis_pos[2]:.3f}) m in the table frame, "
               f"heading {np.degrees(np.arctan2(m.pelvis_R[1, 0], m.pelvis_R[0, 0])):+.1f} deg")
        if sim:  # in simulation the truth is known: how good is perception's calibration?
            from pingpong.deploy.library import quat_mat

            Rt = quat_mat(env.data.qpos[3:7])
            msg += (f" (truth ({env.data.qpos[0]:+.3f}, {env.data.qpos[1]:+.3f}, {env.data.qpos[2]:.3f}), "
                    f"{np.degrees(np.arctan2(Rt[1, 0], Rt[0, 0])):+.1f} deg)")
        logger.info(msg)
    if sim:
        pipeline.policy.reset()
        pipeline.ctrl_manager.reset()
        if sim_ctrl is not None:
            sim_ctrl.start_play()
    elif not a.dry_run:
        ustop = ctrls.get("UnitreeCtrl")
        triggers = dict(getattr(ustop.cfg_ctrl, "triggers", None) or {}) if ustop is not None else {}
        if not any(c in STOP_COMMANDS for c in triggers.values()):
            logger.error("no stop button on the remote (UnitreeCtrl triggers): refusing to apply torque")
            env.shutdown()
            return 1
        try:
            prepare_watching_stops(pipeline, env, triggers, a.prepare_seconds)
        except (StopRequested, KeyboardInterrupt) as e:
            hold_guard(pipeline, env, 1.0 / cfg.policy.freq, str(e) or "Ctrl-C during start-up")
            env.shutdown()
            return 1
    if sim and a.window:
        import mujoco

        def viz():  # the plan the robot is acting on: the racket target (blue, during a swing) and the station (green)
            sw = policy.rt.swing
            out = [dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.09, 0.09, 0.003], pos=[sw.station[0], sw.station[1], 0.004],
                        rgba=[0.1, 0.8, 0.2, 0.6])]
            tp = sw.tp if sw.mode == 1 else np.array([0.0, 0.0, -5.0])
            out.append(dict(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.025] * 3, pos=tp, rgba=[0.2, 0.4, 1.0, 0.6]))
            return out

        env.viz_fn = viz
    monitor = ActionMonitor(env, policy)
    timer = StageTimer()
    timer.wrap(pipeline.policy, "get_observation", "policy.get_observation (planner jobs)")
    timer.wrap(pipeline.policy, "get_pd_target", "policy.get_pd_target (actor)")
    timer.wrap(pipeline.ctrl_manager, "get_ctrl_data", "controllers (mocap, opponent)")
    timer.wrap(env, "step", "env.step (" + ("MuJoCo + ball" if sim else "LowCmd") + ")")
    timer.wrap(pipeline, "step", "whole control step")

    dt, n, n_stale, drops, t_wall0 = 1.0 / cfg.policy.freq, 0, 0, 0, time.perf_counter()
    rec = dict(t=[], pelvis_age=[], mode=[], tts=[], ok=[])
    from collections import deque

    seen = deque(maxlen=a.stale_steps + 3)  # per step: perception messages so far, with the pelvis so far (the watchdog's diagnosis)
    stop_reason = None
    n_reported = 0
    t_next = time.perf_counter()
    try:
        while True:
            pipeline.step(dry_run=a.dry_run)
            n += 1
            info = policy.last_info
            t_now = float(env.data.time) if (sim and not a.dry_run) else time.perf_counter() - t_wall0  # a dry run never advances the simulation
            rec["t"].append(t_now)
            rec["pelvis_age"].append(info.get("pelvis_age", np.inf))
            rec["mode"].append(info.get("mode", -1))
            rec["tts"].append(info.get("time_to_strike", np.nan))
            rec["ok"].append(bool(info.get("ok", False)))
            # ---- watchdogs
            n_stale = n_stale + 1 if (not info.get("ok", False) or info.get("pelvis_age", np.inf) > cfg.policy.pelvis_stale_s) else 0
            q, dq = env.dof_pos, env.dof_vel
            target = policy._hold if hasattr(policy, "_hold") else None
            if target is not None:
                tgt_env = pipeline.policy.actions_adapter.fit(target, template=env.dof_cfg.default_pos)
                if not monitor.check(tgt_env, q, dq):
                    stop_reason = "NaN in the joint targets"
            seen.append((getattr(policy.rt, "n_msgs", getattr(policy.rt, "n_frames", 0)), getattr(policy.rt, "n_pelvis_msgs", -1)))
            if n_stale > a.stale_steps and n > 50:
                d_msgs, d_pel = seen[-1][0] - seen[0][0], seen[-1][1] - seen[0][1]
                age = info.get("pelvis_age", np.inf)
                why = ("NO messages from perception in that time: perception or the relay stopped or stalled (CPU load? a crash?)"
                       if d_msgs == 0 else f"{d_msgs} perception messages, none with the pelvis: Motive lost the pelvis rigid body "
                       "(markers hidden?)" if d_pel == 0 else "(perception's runtime is older: no count of pelvis messages)" if seen[-1][1] < 0
                       else f"{d_pel} pelvis poses arrived, but older than "
                       f"{cfg.policy.pelvis_stale_s * 1000:.0f} ms (the last {age * 1000:.0f} ms): a delay between Motive and here "
                       "(the relay or perception falling behind: CPU load?)")
                stop_reason = f"no fresh pelvis pose for {n_stale} steps ({n_stale * dt * 1000:.0f} ms) - {why}"
            pz = float(info["pelvis"][2]) if "pelvis" in info else None
            if (pz is not None and pz < 0.45) or (sim and env.fallen):
                stop_reason = f"pelvis too low ({pz if pz is not None else float(env.data.qpos[2]):.2f} m): fallen"
            tl = tilt_deg(env.base_quat)
            if tl > a.max_tilt_deg:
                stop_reason = f"pelvis tilted {tl:.0f} deg"
            if getattr(pipeline, "last_commands", None) and any(c in STOP_COMMANDS for c in pipeline.last_commands):
                stop_reason = f"operator stop {pipeline.last_commands}"
            if stop_reason:
                logger.error(f"STOP: {stop_reason}")
                if not a.dry_run and not sim:  # the guard is carried out by stepping: hold it until Ctrl-C
                    hold_guard(pipeline, env, dt, stop_reason)
                break
            if sim and sim_ctrl is not None and sim_ctrl.done:
                for _ in range(25):  # a moment more, so the last ball's outcome is seen through
                    pipeline.step(dry_run=a.dry_run)
                break
            if a.max_seconds and t_now > a.max_seconds:
                logger.info("time limit")
                break
            if sim and not cfg.env.headless and not env.viewer.is_alive:
                logger.info("the window was closed")
                break
            if sim and len(env.results) >= n_reported + 10:  # progress, every 10 balls
                n_reported = len(env.results)
                R = env.results
                last = R[-10:]
                print(f"[play] balls {len(R)}: returned {sum(r['returned'] for r in R) / len(R):.1%} "
                      f"(last 10: {sum(r['returned'] for r in last)}/10), net {sum(r['net'] for r in R)}, no contact {sum(not r['contact'] for r in R)}",
                      flush=True)
            if not sim or a.realtime:
                t_next += dt
                slack = t_next - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    drops += 1
                    t_next = time.perf_counter()
    except KeyboardInterrupt:
        logger.warning("interrupted")
        if not sim and not a.dry_run:
            hold_guard(pipeline, env, dt, "Ctrl-C")
    except Exception as e:  # a bug in the loop with torque on: the guard stop, as for any stop (then the error is shown)
        if not sim and not a.dry_run:
            logger.exception("error in the control loop")
            hold_guard(pipeline, env, dt, f"error: {e!r}")
        raise
    finally:
        wall = time.perf_counter() - t_wall0
        logger.info(f"{n} control steps in {wall:.1f} s wall ({n * dt:.1f} s of control); late steps {drops}")
        timer.report(dt * 1000.0)
        monitor.report()
        if sim:
            R = env.results
            ret = sum(r["returned"] for r in R)
            print(f"\nshots {len(R)}: returned {ret} ({ret / max(len(R), 1):.1%}), net {sum(r['net'] for r in R)}, no contact {sum(not r['contact'] for r in R)}; "
                  f"fell {env.fallen}")
            for r in (R if len(R) <= 16 else []):
                print(f"  ball {r['idx']}: {'RETURNED' if r['returned'] else 'net' if r['net'] else 'missed' if not r['contact'] else 'not returned'}")
            rt = policy.rt
            if perc is not None:
                print(f"perception messages {perc.n_msgs}; lockstep timeouts {perc.n_lockstep_timeouts}")
            print(f"perception messages used {rt.n_frames}; detected shots {len(rt.shots)}; swings: {[s.get('swing') for s in rt.shots]}; "
                  f"shots while busy {rt.n_busy}; dry-run style torque excess steps (joints): {int((monitor.torque_over > 0).sum())}")
        if a.log:
            if a.log.endswith(".json"):
                json.dump(dict(results=env.results if sim else [], shots=policy.rt.shots), open(a.log, "w"), indent=1, default=float)
            else:
                np.savez(a.log, **{k: np.array(v) for k, v in rec.items()})
            print(f"-> {a.log}")
        env.shutdown()


if __name__ == "__main__":
    sys.exit(main())
