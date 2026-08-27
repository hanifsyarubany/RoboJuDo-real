"""[GUARD_STOP]'s overhead head-guard arm pose -- shared by MujocoEnv and UnitreeCppEnv so both
envs use the byte-identical target (never two copies that could silently drift apart).

WHY THIS POSE: G1's model has NO separate head/neck body at all (the kinematic tree stops at
torso_link -- confirmed by enumerating every body in scene_g1_29dof.xml, 2026-08-27), so "protect
the head" can't be computed relative to an actual head frame. This targets a generic, direction-
agnostic failure mode instead -- the torso pitching so the head arcs toward the ground -- rather
than a fall-direction-specific brace (which this codebase has no way to detect at trigger time).

DERIVATION (verified via real forward kinematics on scene_g1_29dof.xml, not guessed):
  1. Probed each arm joint's sign convention individually (nudge +0.5 rad, measure which way the
     wrist moves) to confirm shoulder_pitch NEGATIVE swings the arm forward/up (positive swings it
     backward), shoulder_roll POSITIVE raises/abducts it.
  2. FIRST candidate (superseded, see below) optimized purely for "wrist position relative to the
     shoulder" (compact + high) and found shoulder_pitch=-2.6/roll=0.5/elbow=1.4 -- which DOES land
     the wrist near head height, but does so almost entirely by swinging the SHOULDER back, leaving
     the elbow nearly straight (measured elbow bend: only 10.7 degrees between the upper-arm and
     forearm vectors). Caught by comparing against a real reference photo of a human posing a real
     G1's arm (2026-08-27, user-supplied): the actual desired pose has the elbow clearly bent to
     roughly a right angle, forearm raised roughly PERPENDICULAR to the forward direction, not a
     nearly-straight arm reaching up.
  3. Re-derived with an explicit angle objective instead: computed elbow_bend (angle between the
     shoulder->elbow and elbow->wrist vectors; 0=straight, 90=right-angle, 180=fully folded back)
     and forearm_vs_forward (angle between the forearm vector and the robot's local +x/forward
     axis) via FK, then grid-searched shoulder_pitch/roll/elbow for BOTH near 90 degrees. Found
     shoulder_pitch=-1.5, elbow=-0.15 gives elbow_bend=90.5, forearm_vs_forward=91.1 at roll=0 --
     then swept shoulder_roll upward (0.0 -> 1.0) and confirmed both angles stay ~90-91 degrees
     across that whole range (roll barely affects the bend, only the lateral spread) -- picked
     roll=0.4 for ~22.5cm of wrist-to-wrist clearance (roll=0.0 leaves them only ~10cm apart, a
     tighter self-collision margin than warranted).
  4. Verified the right-arm mirror (only shoulder_roll's sign flips -- matches this policy's own
     default_dof_pos mirroring pattern, e.g. G1UnifiedLocoKickDoF: left/right shoulder_pitch and
     elbow are equal, shoulder_roll is +0.2/-0.2) produces an exact mirror image (residual 0.0000
     across x/y/z).

Result (left arm; right mirrors shoulder_roll's sign only): elbow bent to ~91 degrees (right angle),
forearm ~91 degrees off the forward axis (perpendicular, matching the reference photo), wrist ends
up ~0.45m above torso origin / ~0.2m forward, ~22.5cm of wrist-to-wrist clearance.

THIS IS A REASONED ENGINEERING DEFAULT, NOT AN EMPIRICALLY VALIDATED ONE. There is no official G1
"guard mode" spec to source this from (unlike ESTOP_REAL_DAMPING_KD, which came straight from
unitree_cpp's own source) -- FK-verified means the numbers are internally consistent and collision-
free, not that this is proven to reduce injury on a real fall. Watch it happen in sim before trusting
it on hardware, same as any new real-hardware mechanism in this project.
"""

from __future__ import annotations

import numpy as np

# suffix (joint name minus "left_"/"right_" prefix) -> LEFT-arm target angle, radians.
# shoulder_yaw and every wrist DOF are left at 0 -- they don't materially affect where the forearm
# ends up for THIS pose and zero is the simplest, least-surprising choice absent a specific reason
# to do otherwise.
GUARD_ARM_TARGETS_BY_SUFFIX: dict[str, float] = {
    "shoulder_pitch_joint": -1.5,
    "shoulder_roll_joint": 0.4,  # MIRRORED: right arm uses -0.4
    "shoulder_yaw_joint": 0.0,
    "elbow_joint": -0.15,
    "wrist_roll_joint": 0.0,
    "wrist_pitch_joint": 0.0,
    "wrist_yaw_joint": 0.0,
}
GUARD_MIRRORED_SUFFIXES = frozenset({"shoulder_roll_joint"})

# Moderate, not full, arm authority while guarding -- decisive enough to reliably reach the pose
# against gravity, but not so stiff that an unexpected obstruction (the forearm meeting the ground
# early, contact with equipment) transmits full trained-controller-level force. Not invented
# numbers: GUARD_ARM_KP=20.0 is this project's own G1_29DoF stock WRIST stiffness (used as-is,
# below every arm joint's stock shoulder/elbow value of 40) and GUARD_ARM_KD=2.0 is the stock
# damping already used for every arm joint (shoulder AND wrist alike) -- see
# robojudo/config/g1/env/g1_env_cfg.py::G1_29DoF.
GUARD_ARM_KP = 20.0
GUARD_ARM_KD = 2.0


def build_guard_pose(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """For a given robot's joint-name list (in its own DoF order), returns:
      is_arm_joint: bool array, True at every arm-joint index.
      target_pose: float array (radians), the guard target at arm-joint indices, 0.0 elsewhere
                   (non-arm entries are meaningless -- the caller drives kp->0 there, so no target
                   matters).
    Matches by joint-name SUFFIX after stripping a "left_"/"right_" prefix, not by hardcoded index,
    so this stays correct even if a robot config reorders its DoF list. Joints outside the known
    suffix set (e.g. legs, waist) are simply left False/0.0 -- this function only ever marks
    arm joints whose exact suffix it recognizes, never anything else."""
    n = len(joint_names)
    is_arm = np.zeros(n, dtype=bool)
    target = np.zeros(n, dtype=np.float64)
    for i, name in enumerate(joint_names):
        for side in ("left_", "right_"):
            if not name.startswith(side):
                continue
            suffix = name[len(side) :]
            if suffix not in GUARD_ARM_TARGETS_BY_SUFFIX:
                break
            value = GUARD_ARM_TARGETS_BY_SUFFIX[suffix]
            if side == "right_" and suffix in GUARD_MIRRORED_SUFFIXES:
                value = -value
            is_arm[i] = True
            target[i] = value
            break
    return is_arm, target
