#!/usr/bin/env python3
"""Empirically measure the IMU-derived tilt angle -- RlPipeline.safety_check()'s own
get_gravity_orientation(base_quat) -> arccos(-gz) metric -- reached during NORMAL, non-falling
operation (walking/strafing/turning, an aggressive combined command, a scripted kick) vs. during an
ACTUAL induced fall (a large sideways shove). Used to pick safety_check()'s SAFETY_CHECK_LOCOMOTION_
TILT_RAD / SAFETY_CHECK_KICK_TILT_RAD thresholds (robojudo/pipeline/rl_pipeline.py) with real,
measured margin instead of a guessed number -- see that method's own comment for the numbers this
produced (2026-09-17) and the reasoning behind the task-mode split.

CAVEAT: the "kick" segment here triggers [TRIGGER_KICK] with NO ball present and the robot
standing still -- likely an out-of-distribution swing target relative to a real, properly-aimed
kick (which always has ball-proximity/contact-orientation reward terms shaping it). It topples in
this script's own run. Treat that number as "an upper bound worth being cautious around", not as
"what a real kick's swing normally reaches" -- there's no live-ball harness here to test the latter.

USAGE
  python scripts/measure_fall_tilt.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.utils.util_func import get_gravity_orientation


def axis_for(velocity, row):
    lo, mid, hi = (float(v) for v in row)
    scale_pos, scale_neg = hi - mid, mid - lo
    axis = (velocity - mid) / scale_pos if scale_pos else 0.0
    if axis <= 0:
        axis = (velocity - mid) / scale_neg if scale_neg else 0.0
    return float(np.clip(axis, -1.0, 1.0))


def install_command_source(pipeline):
    """Feed scripted vx/vy/wz through a fake JoystickCtrl axes dict instead of a real device."""
    axes = {"LeftX": 0.0, "LeftY": 0.0, "RightX": 0.0}
    original = pipeline.ctrl_manager.get_ctrl_data

    def get_ctrl_data(env_data):
        data = original(env_data)
        injected = {"JoystickCtrl": {"axes": axes}}
        for key, value in data.items():
            if key not in ("JoystickCtrl", "UnitreeCtrl", "KeyboardCtrl"):
                injected[key] = value
        return injected

    pipeline.ctrl_manager.get_ctrl_data = get_ctrl_data
    return axes


def set_command(axes, policy, vx, vy, wz):
    axes["LeftY"] = axis_for(vx, policy.commands_map[0])
    axes["LeftX"] = axis_for(vy, policy.commands_map[1])
    axes["RightX"] = axis_for(wz, policy.commands_map[2])


def tilt_angle_deg(env):
    gravity_ori = get_gravity_orientation(env.base_quat)
    angle = np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0))
    return float(np.degrees(angle))


def main():
    cfg = ConfigManager(config_name="g1_unified_loco_kick").get_cfg()
    pipeline = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pipeline.env
    env.viewer.close()
    policy = pipeline.policy.policy
    axes = install_command_source(pipeline)
    torso_bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    def reset():
        mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
        mujoco.mj_forward(env.model, env.data)
        env.update()
        pipeline.reset()
        policy.reset()
        set_command(axes, policy, 0.0, 0.0, 0.0)
        for _ in range(int(round(1.0 / pipeline.dt))):
            pipeline.step()

    def run_segment(label, n_s, on_tick=None):
        n_ticks = int(round(n_s / pipeline.dt))
        angles = []
        fell = False
        for t in range(n_ticks):
            if on_tick is not None:
                on_tick(t)
            pipeline.step()
            angles.append(tilt_angle_deg(env))
            if env.data.qpos[2] < 0.4:
                fell = True
                break
        print(
            f"  [{label}] ticks={len(angles)} fell={fell} "
            f"tilt_deg(mean,max)=({np.mean(angles):.1f}, {np.max(angles):.1f}) "
            f"base_z_min={min(env.data.qpos[2], 10):.3f}"
        )
        return angles, fell

    print("== Normal locomotion segments (5s each) ==")
    reset()
    all_normal_angles = []
    for name, cmd in [
        ("forward", (0.6, 0.0, 0.0)),
        ("strafe", (0.0, 0.5, 0.0)),
        ("yaw", (0.0, 0.0, 0.8)),
        ("aggressive combined", (0.6, 0.4, 0.6)),
    ]:
        set_command(axes, policy, *cmd)
        angles, _ = run_segment(name, 5.0)
        all_normal_angles.extend(angles)
        set_command(axes, policy, 0.0, 0.0, 0.0)
        for _ in range(int(round(1.0 / pipeline.dt))):
            pipeline.step()

    print("\n== Scripted kick (NO ball present -- see module docstring's caveat) ==")
    reset()
    set_command(axes, policy, 0.0, 0.0, 0.0)
    policy.post_step_callback(["[TRIGGER_KICK]"])
    kick_angles, _ = run_segment("kick", 3.0)

    print("\n== Induced fall: large lateral shove on torso ==")
    reset()
    set_command(axes, policy, 0.3, 0.0, 0.0)
    for _ in range(int(round(1.0 / pipeline.dt))):
        pipeline.step()

    PUSH_N = 900.0  # a hard shove -- far above body_push_force_max=80N training ever samples
    PUSH_S = 0.15

    def push_tick(t):
        if t < int(round(PUSH_S / pipeline.dt)):
            env.data.xfrc_applied[torso_bid, 1] = PUSH_N  # lateral (body-frame-ish world y) force
        else:
            env.data.xfrc_applied[torso_bid, 1] = 0.0

    fall_angles, fell = run_segment("induced fall", 3.0, on_tick=push_tick)
    env.data.xfrc_applied[torso_bid, :] = 0.0

    print("\n== Summary ==")
    print(f"Max tilt angle across normal locomotion only (walk/strafe/yaw/aggressive): "
          f"{max(all_normal_angles):.1f} deg")
    print(f"Max tilt angle during the scripted (ball-less) kick: {max(kick_angles):.1f} deg "
          "-- see module docstring's caveat before trusting this as a real kick's envelope")
    if fell:
        fall_tick_count = len(fall_angles)
        for cand_deg in (30, 35, 40, 45, 50, 57.3):
            crossing = next((i for i, a in enumerate(fall_angles) if a > cand_deg), None)
            if crossing is not None:
                lead_s = (fall_tick_count - crossing) * pipeline.dt
                print(f"  induced-fall threshold {cand_deg:5.1f} deg: first crossed at tick "
                      f"{crossing:4d}, {lead_s:.2f}s of lead time before qpos[2]<0.4")
            else:
                print(f"  induced-fall threshold {cand_deg:5.1f} deg: never crossed during this fall")
    else:
        print("Induced push did NOT cause a fall (qpos[2] stayed >= 0.4m) -- push too weak, or the "
              "robot recovered on its own.")
        print(f"Max tilt angle reached during the push/recovery: {max(fall_angles):.1f} deg")


if __name__ == "__main__":
    main()
