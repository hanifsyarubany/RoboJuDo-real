"""Unit tests for UnifiedLocoKickPolicy's multi-skill motion selection: reading
skill_motion_start_idx/skill_motion_end_idx ONNX metadata, [TRIGGER_KICK:N] command parsing,
offsetting curr_motion_timestep to the selected skill's own start, capping its per-tick increment
at that skill's own end (never drifting into another skill's embedded frames), and capturing the
correct per-skill yaw offset at trigger time (not always skill 0's, which was the bug this whole
mechanism fixes).

Uses unittest (stdlib) rather than pytest, since the robojudo conda env doesn't have pytest
installed. Isolated via a bare instance (object.__new__, bypassing __init__'s real ONNX-session
load) with a fake `session` object providing just enough of onnxruntime's InferenceSession
interface for _query_ref_quat_at to work deterministically -- no real .onnx file needed, matching
this project's existing holosoma-side bare-instance test convention for the same class of problem.
"""

import sys
import unittest

sys.path.insert(0, "/workspaces/isaaclab_arena/submodules/workspaces/humanoid_deployment/RoboJuDo")

import numpy as np

from robojudo.policy.unified_loco_kick_policy import _TASK_KICK, _TASK_LOCOMOTION, UnifiedLocoKickPolicy


class _FakeSessionInput:
    shape = (1, 10)  # arbitrary obs width; only shape[1] is read


class _FakeSession:
    """ref_quat_xyzw_by_time_step: {time_step: xyzw quat} -- deterministic per-frame lookup so
    tests can assert the exact yaw offset a given skill's start frame should produce."""

    def __init__(self, ref_quat_xyzw_by_time_step: dict[float, np.ndarray]):
        self._table = ref_quat_xyzw_by_time_step

    def get_inputs(self):
        return [_FakeSessionInput()]

    def run(self, output_names, feeds):
        time_step = float(feeds["time_step"][0][0])
        ref_quat = self._table[time_step]
        outs = []
        for name in output_names:
            if name == "ref_quat_xyzw":
                outs.append(ref_quat[None, :])
            elif name in ("joint_pos", "joint_vel"):
                outs.append(np.zeros((1, 29)))
            else:
                raise AssertionError(f"unexpected output requested: {name}")
        return outs


def _identity_quat():
    return np.array([0.0, 0.0, 0.0, 1.0])


def _yaw_quat(yaw_rad: float):
    from scipy.spatial.transform import Rotation as sRot

    return sRot.from_euler("z", yaw_rad).as_quat()


def _make_policy(
    skill_start_idx,
    skill_end_idx,
    ref_quat_table,
    skill_ball_xy=None,
    skill_target_xy=None,
    pre_recovery_idx=None,
    kick_recovery_locomotion_flip_enabled=False,
) -> UnifiedLocoKickPolicy:
    p = object.__new__(UnifiedLocoKickPolicy)
    p.session = _FakeSession(ref_quat_table)
    p._skill_start_idx = skill_start_idx
    p._skill_end_idx = skill_end_idx
    p._skill_ball_xy = skill_ball_xy if skill_ball_xy is not None else []
    p._skill_target_xy = skill_target_xy if skill_target_xy is not None else []
    p._pre_recovery_idx = pre_recovery_idx if pre_recovery_idx is not None else []
    p._kick_recovery_locomotion_flip_enabled = kick_recovery_locomotion_flip_enabled
    p.num_dofs = 29
    p.num_actions = 29
    p.freq = 50.0
    p.reset()
    return p


class TestSkillSelection(unittest.TestCase):
    def test_legacy_metadata_absent_defaults_to_single_skill_starting_at_zero(self):
        p = _make_policy([0], [], {0.0: _identity_quat()})
        self.assertEqual(p._skill_start_idx, [0])
        self.assertEqual(p._skill_end_idx, [])

    def test_plain_trigger_kick_command_selects_skill_zero(self):
        p = _make_policy([0, 432], [432, 847], {0.0: _identity_quat(), 432.0: _identity_quat()})
        p.post_step_callback(["[TRIGGER_KICK]"])
        self.assertEqual(p.kick_skill_id, 0)
        # +1: post_step_callback's own auto-return/advance block runs right after processing the
        # trigger command in this SAME call (pre-existing behavior, not introduced by this change)
        self.assertEqual(p.curr_motion_timestep, 1)

    def test_parameterized_trigger_command_offsets_to_that_skills_start(self):
        p = _make_policy([0, 432], [432, 847], {0.0: _identity_quat(), 432.0: _identity_quat()})
        p.post_step_callback(["[TRIGGER_KICK:1]"])
        self.assertEqual(p.kick_skill_id, 1)
        self.assertEqual(p.curr_motion_timestep, 433)  # skill 1's own start (432) + 1, not skill 0's

    def test_out_of_range_skill_id_falls_back_to_skill_zero(self):
        p = _make_policy([0, 432], [432, 847], {0.0: _identity_quat(), 432.0: _identity_quat()})
        p.post_step_callback(["[TRIGGER_KICK:7]"])
        self.assertEqual(p.kick_skill_id, 0)
        self.assertEqual(p.curr_motion_timestep, 1)

    def test_malformed_skill_id_falls_back_to_skill_zero_without_crashing(self):
        p = _make_policy([0, 432], [432, 847], {0.0: _identity_quat(), 432.0: _identity_quat()})
        p.post_step_callback(["[TRIGGER_KICK:not_a_number]"])
        self.assertEqual(p.kick_skill_id, 0)

    def test_yaw_offset_at_trigger_uses_the_selected_skills_own_start_frame_not_skill_zeros(self):
        # skill 0 starts facing yaw=0, skill 1 starts facing yaw=+90deg -- if the bug this
        # mechanism fixes were still present, triggering skill 1 would wrongly capture skill 0's
        # yaw (0) instead of skill 1's own (pi/2).
        p = _make_policy(
            [0, 100],
            [100, 200],
            {0.0: _yaw_quat(0.0), 100.0: _yaw_quat(np.pi / 2)},
        )
        p.post_step_callback(["[TRIGGER_KICK:1]"])
        self.assertAlmostEqual(p.motion_yaw_offset, np.pi / 2, places=5)

    def test_skill_zero_yaw_offset_reuses_cached_value_byte_identical_to_pre_multi_skill_behavior(self):
        # start_frame == 0 must NOT call _query_ref_quat_at at all (session.run would raise
        # AssertionError on an unexpected output name if it mistakenly queried "joint_pos" etc. at
        # an unplanned time) -- this proves skill 0's path is untouched by the new mechanism.
        p = _make_policy([0], [], {0.0: _yaw_quat(0.3)})
        p.post_step_callback(["[TRIGGER_KICK]"])
        self.assertAlmostEqual(p.motion_yaw_offset, 0.3, places=5)

    def test_curr_motion_timestep_capped_at_selected_skills_own_end_never_drifts_into_next_skill(self):
        p = _make_policy([0, 100], [100, 200], {0.0: _identity_quat(), 100.0: _identity_quat()})
        p.post_step_callback(["[TRIGGER_KICK:0]"])
        p.curr_motion_timestep = 99  # one before skill 0's own last valid frame (100 - 1 = 99)
        p.motion_command_t = np.ones(58)
        p._prev_motion_command_t = np.zeros(58)  # force "still changing" so no auto-return fires
        p.post_step_callback([])
        self.assertEqual(p.curr_motion_timestep, 99)  # capped, did NOT advance into skill 1's [100,200)

    def test_return_to_loco_resets_skill_id_to_zero(self):
        p = _make_policy([0, 432], [432, 847], {0.0: _identity_quat(), 432.0: _identity_quat()})
        p.post_step_callback(["[TRIGGER_KICK:1]"])
        self.assertEqual(p.kick_skill_id, 1)
        p.post_step_callback(["[RETURN_TO_LOCO]"])
        self.assertEqual(p.kick_skill_id, 0)


class TestCycleKickSkill(unittest.TestCase):
    """[CYCLE_KICK_SKILL] advances a PENDING selection (_selected_skill_id) that a later plain
    [TRIGGER_KICK] reads instead of hardcoding skill 0 -- the mechanism behind the deploy
    keyboard's j (cycle) / k (kick) split."""

    def _three_skill_policy(self):
        return _make_policy(
            [0, 100, 300],
            [100, 300, 500],
            {0.0: _identity_quat(), 100.0: _identity_quat(), 300.0: _identity_quat()},
        )

    def test_default_pending_selection_is_skill_zero(self):
        p = self._three_skill_policy()
        self.assertEqual(p._selected_skill_id, 0)

    def test_cycle_advances_pending_selection_without_kicking(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        self.assertEqual(p._selected_skill_id, 1)
        self.assertEqual(p.task_mode, _TASK_LOCOMOTION)
        self.assertEqual(p.kick_skill_id, 0)  # untouched -- meaningless outside an active kick

    def test_cycle_wraps_around_from_last_skill_to_zero(self):
        p = self._three_skill_policy()
        for _ in range(3):
            p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        self.assertEqual(p._selected_skill_id, 0)  # 0 -> 1 -> 2 -> 0

    def test_plain_trigger_kick_uses_the_pending_selection(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])  # now pending = 2
        p.post_step_callback(["[TRIGGER_KICK]"])
        self.assertEqual(p.kick_skill_id, 2)
        self.assertEqual(p.curr_motion_timestep, 301)  # skill 2's own start (300) + 1

    def test_cycle_mid_kick_does_not_affect_the_kick_in_progress(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[TRIGGER_KICK:0]"])
        self.assertEqual(p.kick_skill_id, 0)
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        # currently-kicking skill is untouched; only the pending selection for the NEXT kick moved
        self.assertEqual(p.kick_skill_id, 0)
        self.assertEqual(p._selected_skill_id, 1)

    def test_explicit_trigger_kick_n_also_updates_pending_selection(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[TRIGGER_KICK:2]"])
        self.assertEqual(p._selected_skill_id, 2)
        p.post_step_callback(["[RETURN_TO_LOCO]"])
        p.post_step_callback(["[TRIGGER_KICK]"])  # plain trigger repeats skill 2
        self.assertEqual(p.kick_skill_id, 2)

    def test_pending_selection_survives_return_to_loco(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])  # pending = 1
        p.post_step_callback(["[TRIGGER_KICK]"])
        p.post_step_callback(["[RETURN_TO_LOCO]"])
        self.assertEqual(p.kick_skill_id, 0)  # kick_skill_id resets...
        self.assertEqual(p._selected_skill_id, 1)  # ...but the pending selection sticks

    def test_cycle_is_a_noop_on_single_skill_checkpoint(self):
        p = _make_policy([0], [], {0.0: _identity_quat()})
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        self.assertEqual(p._selected_skill_id, 0)


class TestBallTargetAccessors(unittest.TestCase):
    """get_skill_ball_xy/get_skill_target_xy -- consumed by mujoco_kick_rollout_worker.py to spawn
    the ball and compute kick_target_pos_b at the SELECTED skill's own configured geometry instead
    of a hardcoded, skill-unaware constant."""

    def test_returns_none_when_no_metadata_present(self):
        p = _make_policy([0], [], {0.0: _identity_quat()})
        self.assertIsNone(p.get_skill_ball_xy(0))
        self.assertIsNone(p.get_skill_target_xy(0))

    def test_returns_that_skills_own_values_not_another_skills(self):
        p = _make_policy(
            [0, 432],
            [432, 847],
            {0.0: _identity_quat(), 432.0: _identity_quat()},
            skill_ball_xy=[[7.0, -0.46], [3.5, 0.6]],
            skill_target_xy=[[7.84, -0.46], [8.5, 0.6]],
        )
        self.assertEqual(p.get_skill_ball_xy(0), (7.0, -0.46))
        self.assertEqual(p.get_skill_ball_xy(1), (3.5, 0.6))
        self.assertEqual(p.get_skill_target_xy(0), (7.84, -0.46))
        self.assertEqual(p.get_skill_target_xy(1), (8.5, 0.6))

    def test_out_of_range_skill_id_returns_none(self):
        p = _make_policy(
            [0], [], {0.0: _identity_quat()}, skill_ball_xy=[[7.0, -0.46]], skill_target_xy=[[7.84, -0.46]]
        )
        self.assertIsNone(p.get_skill_ball_xy(5))
        self.assertIsNone(p.get_skill_target_xy(-1))


class TestKickRecoveryLocomotionFlipAutoReturn(unittest.TestCase):
    """kick_recovery_locomotion_flip_enabled (Stage D) -- when set, an explicit crossing
    (curr_motion_timestep reaching self._pre_recovery_idx, the same boundary training flips
    task_mode at: time_steps reaching pre_recovery_motion_end_idx, the end of the WHOLE
    authored/raw clip) supersedes the plateau heuristic entirely, so a Stage D checkpoint returns
    to locomotion at the SAME instant training flipped its task_mode -- once genuine captured
    motion runs out, before the appended synthetic recovery-transition + hold tail, not ~2s later
    once a plateau-detection pass would have noticed. (2026-08-10: settled on this boundary after
    two other same-day experiments -- stand_start_idx, swing end, too narrow; motion_end_idx, the
    clip's true last frame, too wide -- were both live-tested in RoboJuDo and superseded.)"""

    def test_legacy_checkpoint_defaults_to_flag_disabled(self):
        p = _make_policy([0], [], {0.0: _identity_quat()})
        self.assertFalse(p._kick_recovery_locomotion_flip_enabled)
        self.assertEqual(p._pre_recovery_idx, [])

    def test_crossing_pre_recovery_idx_triggers_auto_return_when_enabled(self):
        p = _make_policy(
            [0], [], {0.0: _identity_quat()}, pre_recovery_idx=[300], kick_recovery_locomotion_flip_enabled=True
        )
        p.post_step_callback(["[TRIGGER_KICK]"])
        self.assertEqual(p.task_mode, _TASK_KICK)
        p.curr_motion_timestep = 300  # exactly at the boundary
        p.post_step_callback([])
        self.assertEqual(p.task_mode, _TASK_LOCOMOTION)

    def test_not_yet_reaching_the_boundary_stays_in_kick_mode(self):
        p = _make_policy(
            [0], [], {0.0: _identity_quat()}, pre_recovery_idx=[300], kick_recovery_locomotion_flip_enabled=True
        )
        p.post_step_callback(["[TRIGGER_KICK]"])
        p.curr_motion_timestep = 299  # one tick short of the boundary
        p.post_step_callback([])
        self.assertEqual(p.task_mode, _TASK_KICK)

    def test_uses_the_triggered_skills_own_boundary_not_skill_zeros(self):
        p = _make_policy(
            [0, 100],
            [100, 200],
            {0.0: _identity_quat(), 100.0: _identity_quat()},
            pre_recovery_idx=[80, 180],
            kick_recovery_locomotion_flip_enabled=True,
        )
        p.post_step_callback(["[TRIGGER_KICK:1]"])
        p.curr_motion_timestep = 80  # crosses skill 0's boundary (80) but NOT skill 1's (180)
        p.post_step_callback([])
        self.assertEqual(p.task_mode, _TASK_KICK, "must check skill 1's own boundary (180), not skill 0's (80)")
        p.curr_motion_timestep = 180
        p.post_step_callback([])
        self.assertEqual(p.task_mode, _TASK_LOCOMOTION)

    def test_flag_disabled_ignores_pre_recovery_idx_and_falls_back_to_plateau(self):
        """Regression guard for the exact bug this design avoids: pre_recovery metadata is
        exported for EVERY kick-capable checkpoint, Stage D or not -- gating on the flag (not just
        "boundary metadata present") is what keeps a non-Stage-D checkpoint's behavior byte-
        identical to before Stage D deployment support existed."""
        p = _make_policy(
            [0], [], {0.0: _identity_quat()}, pre_recovery_idx=[300], kick_recovery_locomotion_flip_enabled=False
        )
        p.post_step_callback(["[TRIGGER_KICK]"])
        p.curr_motion_timestep = 300  # would cross the Stage D boundary if the flag were honored
        p.motion_command_t = np.zeros(58)
        p._prev_motion_command_t = np.ones(58)  # NOT plateaued -- plateau heuristic must not fire either
        p.post_step_callback([])
        self.assertEqual(p.task_mode, _TASK_KICK, "flag off must ignore pre_recovery_idx entirely")

    def test_absent_metadata_falls_back_to_plateau_heuristic_unchanged(self):
        # regression guard: flag unset (pre-Stage-D checkpoint) must still auto-return via the old
        # plateau detection, byte-identical to behavior before this mechanism existed.
        p = _make_policy([0], [], {0.0: _identity_quat()})  # flag/pre_recovery_idx default to False/[]
        p.post_step_callback(["[TRIGGER_KICK]"])
        p.curr_motion_timestep = int(5 * p.freq)  # past the min-elapsed floor
        p.motion_command_t = np.ones(58)
        p._prev_motion_command_t = np.ones(58)  # "plateaued" this tick
        p._kick_hold_ticks = int(3 * p.freq) - 1  # one tick short of the hold threshold
        p.post_step_callback([])
        self.assertEqual(p.task_mode, _TASK_LOCOMOTION)


if __name__ == "__main__":
    unittest.main()
