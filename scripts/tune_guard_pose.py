"""Standalone tuning tool for [GUARD_STOP]'s overhead head-guard arm pose
(robojudo/environment/utils/guard_pose.py) -- the same forward-kinematics methodology used to
derive and correct that pose, packaged so you can iterate on it directly instead of asking an
agent to re-run one-off snippets each time.

WHAT IT DOES: loads the real holosoma G1 MuJoCo model (no viewer, headless, ~instant), sets the
LEFT arm's 4 pose-relevant joints (shoulder_pitch/roll/yaw, elbow) to either guard_pose.py's
current values or your CLI overrides, mirrors to the right arm using the exact same rule
build_guard_pose() uses (only shoulder_roll's sign flips), and reports the numbers that matter:

  elbow_bend         : angle between the shoulder->elbow and elbow->wrist vectors.
                        0 = perfectly straight arm, 90 = right-angle bend, 180 = folded back on
                        itself. This is what "the elbow is bent ~90 degrees" in the pose's
                        docstring actually measures -- NOT the raw elbow joint value, which does
                        NOT correspond 1:1 to this (a nearly-straight arm and a sharply-bent one
                        can use very different raw elbow values depending on the shoulder angle --
                        this bit a previous iteration of this pose, see guard_pose.py's history).
  forearm_vs_forward  : angle between the forearm vector (elbow->wrist) and the robot's local
                        +x/forward axis. 90 = perpendicular to forward (a vertical/lateral
                        forearm), 0 = pointing straight forward, 180 = pointing straight back.
  wrist_rel_torso     : wrist position relative to torso_link's origin, in the robot's own frame
                        (x=forward, y=left, z=up). There's no head body in this model at all (see
                        guard_pose.py) so there's no single "distance to head" number -- judge
                        height (z) and forward reach (x) against your own sense of where the head
                        actually is (roughly 0.25-0.35m above torso origin for this robot, per
                        prior FK exploration).
  wrist_to_wrist / elbow_to_elbow : straight-line clearance between the two arms. Below ~15cm is
                        worth a visual check for self-collision risk; below ~5cm treat as unsafe.
  joint limit check   : flags any target outside that joint's own position_limits (G1_29DoF).

USAGE
  # Report the CURRENT guard_pose.py values (no args):
  python scripts/tune_guard_pose.py

  # Try a candidate without touching guard_pose.py yet:
  python scripts/tune_guard_pose.py --shoulder-pitch -1.3 --shoulder-roll 0.5 --elbow -0.1

  # Only override what you want to change -- omitted flags keep guard_pose.py's current value:
  python scripts/tune_guard_pose.py --elbow 0.1

Once a candidate looks right here, this script prints the exact dict snippet to paste into
GUARD_ARM_TARGETS_BY_SUFFIX in guard_pose.py -- it does NOT edit that file for you (a pose change
is worth a deliberate, reviewed edit, not a silent auto-write).

THIS IS NOT THE LAST STEP. These numbers describe the IDEALIZED static pose (robot standing still,
arms snapped straight there). The real behavior also depends on GUARD_ARM_KP/KD (how firmly/fast
the arms actually track there against gravity and a simultaneously-collapsing body -- also in
guard_pose.py) and GUARD_STOP_ARM_RAMP_SECONDS (robojudo/pipeline/rl_pipeline.py, how long the
transition takes). After picking numbers here, watch it for real:
    conda activate robojudo && python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick
    (press 'g' to trigger [GUARD_STOP], '`' to reset and try again)
That live check is what actually validates the pose -- this script only tells you the geometry is
internally consistent (collision-free, within joint limits, the angles you asked for), not that it
looks right or reaches the target fast enough under real dynamics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from robojudo.config.g1.env.g1_env_cfg import G1_29DoF
from robojudo.environment.utils.guard_pose import (
    GUARD_ARM_KD,
    GUARD_ARM_KP,
    GUARD_ARM_TARGETS_BY_SUFFIX,
    GUARD_MIRRORED_SUFFIXES,
)

_SCENE_XML = (
    Path(__file__).resolve().parent.parent / "assets/robots/g1/holosoma_model/scene_g1_29dof.xml"
).as_posix()

_SUFFIXES = ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint"]
_CLI_NAMES = {  # suffix -> argparse dest
    "shoulder_pitch_joint": "shoulder_pitch",
    "shoulder_roll_joint": "shoulder_roll",
    "shoulder_yaw_joint": "shoulder_yaw",
    "elbow_joint": "elbow",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for suffix, cli_name in _CLI_NAMES.items():
        ap.add_argument(
            f"--{cli_name.replace('_', '-')}",
            dest=cli_name,
            type=float,
            default=None,
            help=f"Override the LEFT arm's {suffix} target (radians). Default: guard_pose.py's "
            f"current value ({GUARD_ARM_TARGETS_BY_SUFFIX[suffix]}). Right arm mirrors automatically "
            f"({'sign-flipped' if suffix in GUARD_MIRRORED_SUFFIXES else 'same value'}).",
        )
    return ap.parse_args()


def _joint_limits() -> dict[str, tuple[float, float]]:
    dof = G1_29DoF()
    return dict(zip(dof.joint_names, dof.position_limits))


def main() -> int:
    args = parse_args()

    left_pose = dict(GUARD_ARM_TARGETS_BY_SUFFIX)
    for suffix, cli_name in _CLI_NAMES.items():
        override = getattr(args, cli_name)
        if override is not None:
            left_pose[suffix] = override

    model = mujoco.MjModel.from_xml_path(_SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    def qadr(name: str) -> int:
        return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

    def body_pos(name: str) -> np.ndarray:
        return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    limits = _joint_limits()
    limit_issues = []
    for side in ("left", "right"):
        for suffix, value in left_pose.items():
            if side == "right" and suffix in GUARD_MIRRORED_SUFFIXES:
                value = -value
            joint_name = f"{side}_{suffix}"
            data.qpos[qadr(joint_name)] = value
            lo, hi = limits[joint_name]
            if not (lo <= value <= hi):
                limit_issues.append(f"{joint_name}={value:.3f} OUTSIDE limit [{lo:.3f}, {hi:.3f}]")
    mujoco.mj_forward(model, data)

    torso = body_pos("torso_link")
    fwd = np.array([1.0, 0.0, 0.0])
    print(f"{'joint':<24}{'left':>10}{'right':>10}")
    for suffix, value in left_pose.items():
        right_value = -value if suffix in GUARD_MIRRORED_SUFFIXES else value
        print(f"{suffix:<24}{value:>10.3f}{right_value:>10.3f}")
    print()

    wrists = {}
    for side in ("left", "right"):
        shoulder = body_pos(f"{side}_shoulder_pitch_link")
        elbow = body_pos(f"{side}_elbow_link")
        wrist = body_pos(f"{side}_wrist_yaw_link")
        wrists[side] = wrist
        ua, fa = elbow - shoulder, wrist - elbow
        ua_n, fa_n = ua / np.linalg.norm(ua), fa / np.linalg.norm(fa)
        elbow_bend = np.degrees(np.arccos(np.clip(np.dot(ua_n, fa_n), -1, 1)))
        forearm_vs_fwd = np.degrees(np.arccos(np.clip(np.dot(fa_n, fwd), -1, 1)))
        rel_torso = wrist - torso
        rel_shoulder = wrist - shoulder
        print(
            f"{side:>5}: elbow_bend={elbow_bend:6.1f}deg  forearm_vs_forward={forearm_vs_fwd:6.1f}deg  "
            f"wrist_rel_torso=({rel_torso[0]:+.3f}, {rel_torso[1]:+.3f}, {rel_torso[2]:+.3f})  "
            f"height_above_shoulder={rel_shoulder[2]:+.3f}"
        )

    wrist_wrist = np.linalg.norm(wrists["left"] - wrists["right"])
    left_elbow = body_pos("left_elbow_link")
    right_elbow = body_pos("right_elbow_link")
    elbow_elbow = np.linalg.norm(left_elbow - right_elbow)
    print(f"\nwrist-to-wrist clearance: {wrist_wrist:.3f} m   elbow-to-elbow clearance: {elbow_elbow:.3f} m")
    if wrist_wrist < 0.05 or elbow_elbow < 0.05:
        print("  !! under 5cm -- likely self-collision, reconsider this pose")
    elif wrist_wrist < 0.15 or elbow_elbow < 0.15:
        print("  ~ under 15cm -- give it a visual check in the live viewer before trusting it")

    if limit_issues:
        print("\n!! JOINT LIMIT VIOLATIONS:")
        for issue in limit_issues:
            print(f"  {issue}")
    else:
        print("\njoint limits: OK (all targets within range)")

    print("\nGUARD_ARM_KP / GUARD_ARM_KD (unchanged by this script):", GUARD_ARM_KP, "/", GUARD_ARM_KD)

    print("\nTo make this the guard pose, paste into GUARD_ARM_TARGETS_BY_SUFFIX in")
    print("robojudo/environment/utils/guard_pose.py:")
    print("{")
    for suffix, value in left_pose.items():
        marker = "  # MIRRORED: right arm uses " + f"{-value:.3f}" if suffix in GUARD_MIRRORED_SUFFIXES else ""
        print(f'    "{suffix}": {value:.3f},{marker}')
    print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
