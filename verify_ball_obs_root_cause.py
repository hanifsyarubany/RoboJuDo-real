#!/usr/bin/env python3
"""One-off root-cause verification (2026-07-17): does feeding the Stage-C unified loco+kick
RoboJuDo policy a REAL ball/target observation (matching training's exact formula) instead of the
hardcoded zero (unified_loco_kick_policy.py:301-302) fix the sim2sim kick fall?

Uses scene_g1_29dof_with_ball.xml (a real, physically-simulated ball, same geom params as
holosoma's own MuJoCo backend -- see that file's docstring) so BOTH trials below share the exact
same physics; the ONLY variable is whether kick_ball_pos_b/kick_target_pos_b are hardcoded zero
(current RoboJuDo behavior) or real, computed every tick from the live ball body + robot root,
using training's own formula (managers/observation/terms/unified.py::ball_pos_b/target_pos_b):
  rel = ball_pos_w - robot_root_pos_w                       (full 3D, real z)
  ball_pos_b = quat_rotate_inverse(yaw_quat(base_quat), rel)  (heading frame: yaw-only rotation)
target_pos_b is the same transform, xy-only, against a fixed world target point.

Run: /workspaces/isaaclab_arena/submodules/workspaces/conda_env/robojudo/bin/python \
     verify_ball_obs_root_cause.py --onnx-path <path> --output-dir <dir>
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROBOJUDO_REPO = str(Path(__file__).parent)
SCENE_WITH_BALL = str(Path(__file__).parent / "assets/robots/g1/holosoma_model/scene_g1_29dof_with_ball.xml")

BALL_WORLD_POS = np.array([1.3, 0.0, 0.11])
TARGET_WORLD_POS = np.array([1.3 + 2.8, 0.0 - 0.46])  # matches training's ball->target relative offset


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--settle-s", type=float, default=1.5)
    p.add_argument("--hold-s", type=float, default=8.0)
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    return p.parse_args()


def run_trial(cfg_onnx_path: str, out_mp4: str, real_ball_obs: bool, width: int, height: int, fps: int, settle_s: float, hold_s: float) -> dict:
    sys.path.insert(0, ROBOJUDO_REPO)
    import mujoco

    import robojudo.config.g1  # noqa: F401 -- registers config
    import robojudo.pipeline  # noqa: F401 -- registers pipeline
    from robojudo.config import cfg_registry
    from robojudo.policy.unified_loco_kick_policy import _TASK_KICK
    from robojudo.utils.util_func import calc_heading_quat_np, quat_rotate_inverse_np

    cfg = cfg_registry.get("g1_unified_loco_kick")()
    cfg.policy.onnx_path = cfg_onnx_path
    cfg.env.xml = SCENE_WITH_BALL  # ball-augmented scene, SAME for both trials
    pl = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pl.env
    env.viewer.is_alive = False
    inner = pl.policy.policy
    inner._update_velocity_command = lambda cd: None

    # Resolve the ball's freejoint qpos address directly from the compiled model (same pattern as
    # holosoma's own MuJoCo._set_ball_joint_addressing).
    ball_jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
    assert ball_jid != -1, "ball_freejoint not found in compiled model"
    ball_qpos_addr = int(env.model.jnt_qposadr[ball_jid])

    z_trace = []
    obs_log = []

    if real_ball_obs:
        # Wrap get_observation: call the original (which hardcodes kick_ball_pos_b/kick_target_pos_b
        # to zero), then overwrite those exact slots with the REAL, live-computed values using
        # training's own heading-frame formula. Alphabetical obs layout (see policy docstring):
        # kick_actions[0:29] kick_ball_pos_b[29:32] kick_base_ang_vel[32:35] kick_dof_pos[35:64]
        # kick_dof_vel[64:93] kick_motion_command[93:151] kick_motion_ref_ori_b[151:157]
        # kick_target_pos_b[157:159] loco_*[159:261]
        orig_get_observation = inner.get_observation

        def patched_get_observation(env_data, ctrl_data):
            obs, extras = orig_get_observation(env_data, ctrl_data)
            # Training tags kick_ball_pos_b/kick_target_pos_b task_mode="kick" -- the observation
            # MANAGER zeros them during locomotion mode regardless of what ball_pos_b() itself
            # computes (see config_values/unified/g1/observation.py:46-56). Replicate that masking
            # exactly: only substitute real values while the policy is actually in kick mode,
            # otherwise leave the original (already-zero) slots untouched.
            if inner.task_mode == _TASK_KICK:
                ball_pos_w = env.data.qpos[ball_qpos_addr : ball_qpos_addr + 3].copy()
                robot_pos_w = env.base_pos.copy()
                base_quat_xyzw = env.base_quat.copy()  # RoboJuDo convention: xyzw

                heading_quat = calc_heading_quat_np(base_quat_xyzw)
                rel_ball = ball_pos_w - robot_pos_w
                ball_pos_b = quat_rotate_inverse_np(heading_quat, rel_ball)

                rel_target = np.array([TARGET_WORLD_POS[0] - robot_pos_w[0], TARGET_WORLD_POS[1] - robot_pos_w[1], 0.0])
                target_pos_b = quat_rotate_inverse_np(heading_quat, rel_target)[:2]

                obs[29:32] = ball_pos_b.astype(np.float32)
                obs[157:159] = target_pos_b.astype(np.float32)
            obs_log.append((float(obs[29]), float(obs[30]), float(obs[31]), float(obs[157]), float(obs[158])))
            return obs, extras

        inner.get_observation = patched_get_observation

    try:
        renderer = mujoco.Renderer(env.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        camera.distance = 4.0
        camera.azimuth = 90.0
        camera.elevation = -5.0

        mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
        mujoco.mj_forward(env.model, env.data)
        env.update()

        stderr_path = out_mp4 + ".ffmpeg_stderr.log"
        with open(stderr_path, "wb") as stderr_f:
            proc = subprocess.Popen(
                [
                    shutil.which("ffmpeg") or "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(fps),
                    "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                    out_mp4,
                ],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_f,
            )

            def capture_frame():
                camera.lookat[:] = [env.data.qpos[0], env.data.qpos[1], 0.6]
                renderer.update_scene(env.data, camera=camera)
                proc.stdin.write(renderer.render().tobytes())

            def step_zero_vel():
                inner.lin_vel_command = np.zeros(2)
                inner.ang_vel_command = 0.0
                pl.step()
                z_trace.append(float(env.data.qpos[2]))

            try:
                for _ in range(int(settle_s * fps)):
                    step_zero_vel()
                    capture_frame()

                env.update()
                inner.get_observation(env.get_data(), {})
                inner.post_step_callback(["[TRIGGER_KICK]"])

                for _ in range(int(hold_s * fps)):
                    step_zero_vel()
                    capture_frame()
            finally:
                proc.stdin.close()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
        rc = proc.returncode
    finally:
        pl.env.shutdown()

    return {
        "returncode": rc,
        "min_z": min(z_trace) if z_trace else None,
        "final_z": z_trace[-1] if z_trace else None,
        "z_trace_sample": z_trace[:: max(1, len(z_trace) // 20)],
        "obs_log_sample": obs_log[:: max(1, len(obs_log) // 10)] if obs_log else [],
    }


def main():
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== TRIAL A: baseline (ball physically present, observation hardcoded ZERO) ===", flush=True)
    res_a = run_trial(args.onnx_path, str(out_dir / "trial_a_zero_obs.mp4"), real_ball_obs=False,
                       width=args.width, height=args.height, fps=args.fps, settle_s=args.settle_s, hold_s=args.hold_s)
    print("TRIAL A RESULT:", res_a, flush=True)

    print("=== TRIAL B: fixed (real ball/target observation, training formula) ===", flush=True)
    res_b = run_trial(args.onnx_path, str(out_dir / "trial_b_real_obs.mp4"), real_ball_obs=True,
                       width=args.width, height=args.height, fps=args.fps, settle_s=args.settle_s, hold_s=args.hold_s)
    print("TRIAL B RESULT:", res_b, flush=True)

    print("=== SUMMARY ===", flush=True)
    print(f"Trial A (zero obs)  min_z={res_a['min_z']:.3f}  final_z={res_a['final_z']:.3f}", flush=True)
    print(f"Trial B (real obs)  min_z={res_b['min_z']:.3f}  final_z={res_b['final_z']:.3f}", flush=True)


if __name__ == "__main__":
    main()
