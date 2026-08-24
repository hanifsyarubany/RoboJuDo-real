#!/usr/bin/env python3
"""Open-loop physics divergence check (2026-07-17): replay the EXACT PD-target sequence recorded
from a falling IsaacSim/PhysX rollout directly into MuJoCo (no policy involved on the MuJoCo side
at all -- just the same commanded joint targets, applied via the identical PD law, starting from
the identical initial state). Any divergence between the two resulting trajectories is then purely
attributable to the physics engine itself, not to the policy reacting differently to different
observations -- eliminates the closed-loop-compounding confound.

Frame conventions handled explicitly:
  - position: world frame, direct transfer.
  - quaternion: holosoma/IsaacSim root_states are xyzw; MuJoCo's native qpos[3:7] is wxyz ->
    reordered on the way in.
  - linear velocity: world frame in both -> direct transfer.
  - angular velocity: holosoma root_states ang_vel is WORLD frame; MuJoCo's free-joint qvel[3:6]
    is BODY frame -> rotated by the inverse of the base orientation on the way in.
"""
import pickle
import sys

import numpy as np

ROBOJUDO_REPO = "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo"
sys.path.insert(0, ROBOJUDO_REPO)


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def world_to_body_angvel(quat_xyzw, ang_vel_world):
    from scipy.spatial.transform import Rotation as sRot

    r = sRot.from_quat(quat_xyzw)
    return r.inv().apply(ang_vel_world)


def main():
    with open("/tmp/isaacsim_survivor_recording.pkl", "rb") as f:
        rec = pickle.load(f)

    pd_full = rec["pd_target_sequence"]  # (604, 29), 4 substeps per control tick
    traj = rec["trajectory"]  # list of 151 dicts (t=0 .. t=150)
    # The leading 4-row group belongs to trigger_kick()'s internal reset_envs_idx (which computes
    # torques once for the reset itself), not the recording loop's first real step -- drop it so
    # the remaining groups align 1:1 with traj[1:].
    pd_seq = pd_full[4::4]  # (150, 29) -- one PD target per control tick, aligned to traj[1:]
    assert len(pd_seq) == len(traj) - 1, f"{len(pd_seq)} vs {len(traj)-1}"

    dof_names_isaac = rec["dof_names"]

    from robojudo.config.g1.env.g1_holosoma_env_cfg import G1HolosomaMujocoEnvCfg
    from robojudo.environment.mujoco_env import MujocoEnv

    env = MujocoEnv(cfg_env=G1HolosomaMujocoEnvCfg())
    env.viewer.is_alive = False

    # Confirm RoboJuDo's own dof order matches the recorded IsaacSim order (both ultimately derive
    # from the same holosoma robot config, but verify rather than assume).
    dof_names_rj = env.cfg_env.dof.joint_names
    print("===DIAG=== isaac dof order:", dof_names_isaac, flush=True)
    print("===DIAG=== robojudo dof order:", dof_names_rj, flush=True)
    assert list(dof_names_isaac) == list(dof_names_rj), "DOF ORDER MISMATCH -- must remap before replay"
    print("===DIAG=== DOF order matches exactly, no remap needed", flush=True)

    # CRITICAL: env.stiffness/env.damping default to G1HolosomaDoF's own static config values,
    # NOT the training/ONNX-derived gains the real policy uses (those are loaded from ONNX
    # metadata by UnifiedLocoKickPolicy.__init__, which this script bypasses entirely since it
    # drives the raw MujocoEnv directly). Caught via this exact print: default stiffness[:6] was
    # [100,100,100,150,40,40] vs the recorded training gains [40.18,99.10,40.18,99.10,28.50,28.50]
    # -- an entirely different, much stiffer robot. Must override explicitly before replay.
    print("===DIAG=== isaac p_gains[:6]:", rec["p_gains"][:6], flush=True)
    print("===DIAG=== robojudo DEFAULT stiffness[:6] (WRONG, about to override):", env.stiffness[:6], flush=True)
    env.stiffness = np.asarray(rec["p_gains"], dtype=np.float32)
    env.damping = np.asarray(rec["d_gains"], dtype=np.float32)
    print("===DIAG=== robojudo stiffness[:6] AFTER override:", env.stiffness[:6], flush=True)
    print("===DIAG=== robojudo torque_limits[:6]:", env.torque_limits[:6], flush=True)
    if "torque_limits" in rec:
        print("===DIAG=== isaac torque_limits[:6]:", rec["torque_limits"][:6], flush=True)

    # ---- Set MuJoCo's initial state to EXACTLY match IsaacSim's state at kick-trigger (t=0) ----
    t0 = traj[0]
    import mujoco

    mujoco.mj_resetData(env.model, env.data)
    env.data.qpos[0:3] = t0["base_pos"]
    env.data.qpos[3:7] = quat_xyzw_to_wxyz(t0["base_quat"])
    env.data.qpos[7 : 7 + env.num_dofs] = t0["dof_pos"]
    env.data.qvel[0:3] = t0["base_lin_vel"]
    env.data.qvel[3:6] = world_to_body_angvel(t0["base_quat"], t0["base_ang_vel"])
    env.data.qvel[6 : 6 + env.num_dofs] = t0["dof_vel"]
    mujoco.mj_forward(env.model, env.data)
    env.update()

    print("===DIAG=== initial state set. base_pos:", env.base_pos, "vs isaac:", t0["base_pos"], flush=True)
    print("===DIAG=== initial dof_pos[:6]:", env.dof_pos[:6], "vs isaac:", t0["dof_pos"][:6], flush=True)

    # ---- Replay the exact PD-target sequence, open loop, recording the resulting trajectory ----
    mj_traj = [
        {
            "dof_pos": env.dof_pos.copy(),
            "base_pos": env.base_pos.copy(),
            "base_quat_xyzw": env.base_quat.copy(),
        }
    ]
    for step, pd_target in enumerate(pd_seq):
        env.step(pd_target.astype(np.float32))
        env.update()
        mj_traj.append(
            {
                "dof_pos": env.dof_pos.copy(),
                "base_pos": env.base_pos.copy(),
                "base_quat_xyzw": env.base_quat.copy(),
            }
        )
        if step % 25 == 0:
            print(f"===DIAG=== step={step} mujoco_z={env.base_pos[2]:.4f} isaac_z={traj[step+1]['base_pos'][2]:.4f}", flush=True)

    # ---- Compare per-step: base height + per-joint position error ----
    print("===DIAG=== ---- COMPARISON (per control tick) ----", flush=True)
    print(f"{'step':>5} {'t(s)':>6} {'isaac_z':>8} {'mj_z':>8} {'|dz|':>7} {'max_joint_err(rad)':>19} {'worst_joint':>28}", flush=True)
    first_divergence_step = None
    for i in range(len(mj_traj)):
        iso_z = traj[i]["base_pos"][2]
        mj_z = mj_traj[i]["base_pos"][2]
        dz = abs(iso_z - mj_z)
        joint_err = np.abs(traj[i]["dof_pos"] - mj_traj[i]["dof_pos"])
        worst_idx = int(joint_err.argmax())
        worst_val = joint_err[worst_idx]
        worst_name = dof_names_isaac[worst_idx]
        if i % 10 == 0 or (first_divergence_step is None and (dz > 0.05 or worst_val > 0.15)):
            print(f"{i:5d} {i/50:6.2f} {iso_z:8.4f} {mj_z:8.4f} {dz:7.4f} {worst_val:19.4f} {worst_name:>28}", flush=True)
        if first_divergence_step is None and (dz > 0.05 or worst_val > 0.15):
            first_divergence_step = i
    print(f"===DIAG=== FIRST significant divergence (|dz|>0.05m or joint_err>0.15rad) at step={first_divergence_step} (t={first_divergence_step/50 if first_divergence_step else None:.3f}s)" if first_divergence_step else "===DIAG=== no significant divergence detected in window", flush=True)

    with open("/tmp/mujoco_replay_survivor_trajectory.pkl", "wb") as f:
        pickle.dump({"mj_traj": mj_traj, "isaac_traj": traj, "dof_names": dof_names_isaac}, f)
    print("===DIAG=== saved comparison data", flush=True)


if __name__ == "__main__":
    main()
