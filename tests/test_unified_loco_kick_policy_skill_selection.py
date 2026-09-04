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
    ready_gesture_enabled=False,
    ready_gesture_user_on=False,
    skill_ball_halfwidth_xy=None,
    ready_gesture_only_when_standing=True,
    skill_cycle_gesture_enabled=False,
    manual_kick_aim_enabled=False,
    manual_kick_aim_step_deg=5.0,
    skill_kick_aim_max_deg=None,
    kick_aim_theta_ref_deg=45.0,
    autonav_enabled=False,
    autonav_kp_approach=1.5,
    autonav_max_speed=0.35,
    autonav_kp_yaw=2.0,
    autonav_manual_deadzone=0.05,
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
    # readiness-gesture attrs normally set in __init__ (bypassed here by object.__new__)
    p._ready_gesture_enabled = ready_gesture_enabled
    p._ready_gesture_ramp_s = 0.4
    p._ready_gesture_shoulder_amp_rad = 0.5
    p._ready_gesture_elbow_amp_rad = 0.6
    p._ready_gesture_freq_hz = 1.2
    p._ready_gesture_only_when_standing = ready_gesture_only_when_standing
    p._skill_ball_halfwidth_xy = (
        skill_ball_halfwidth_xy
        if skill_ball_halfwidth_xy is not None
        else [[0.1, 0.1] for _ in range(len(skill_start_idx))]
    )
    p._right_shoulder_pitch_idx = 22  # right_shoulder_pitch_joint in the standard G1 dof order
    p._right_elbow_idx = 25  # right_elbow_joint
    # skill-cycled-gesture attrs normally set in __init__ (bypassed here by object.__new__)
    p._skill_cycle_gesture_enabled = skill_cycle_gesture_enabled
    p._skill_cycle_gesture_duration_s = 0.6
    p._skill_cycle_gesture_shoulder_amp_rad = 0.55
    p._skill_cycle_gesture_elbow_amp_rad = 0.45
    p._skill_cycle_gesture_swings = 1.0
    p._left_shoulder_pitch_idx = 15  # left_shoulder_pitch_joint in the standard G1 dof order
    p._left_elbow_idx = 18  # left_elbow_joint
    # manual-kick_aim_theta attrs normally set in __init__ (bypassed here by object.__new__)
    p._manual_kick_aim_enabled = manual_kick_aim_enabled
    p._manual_kick_aim_step_deg = manual_kick_aim_step_deg
    p._skill_kick_aim_max_deg = (
        skill_kick_aim_max_deg if skill_kick_aim_max_deg is not None else [None for _ in range(len(skill_start_idx))]
    )
    p._kick_aim_theta_ref_deg = kick_aim_theta_ref_deg
    # autonav attrs normally set in __init__ (bypassed here by object.__new__)
    p._autonav_enabled = autonav_enabled
    p._autonav_kp_approach = autonav_kp_approach
    p._autonav_max_speed = autonav_max_speed
    p._autonav_kp_yaw = autonav_kp_yaw
    p._autonav_manual_deadzone = autonav_manual_deadzone
    # commands_map / rate-limit attrs normally set in __init__ -- large rate limits so a single
    # _update_velocity_command call converges self._smoothed_cmd to the target in one tick, keeping
    # tests simple (matches G1UnifiedLocoKickPolicyCfg's default commands_map magnitudes).
    p.commands_map = [np.array([-0.8, 0.0, 0.8]), np.array([0.5, 0.0, -0.5]), np.array([0.8, 0.0, -0.8])]
    p._cmd_max_mag = np.array([0.8, 0.5, 0.8])
    p._cmd_rate_limit_per_tick = np.array([1000.0, 1000.0, 1000.0])
    p._cmd_decel_limit_per_tick = np.array([1000.0, 1000.0, 1000.0])
    p._cmd_zero_snap = 0.02
    p.zero_cmd_eps = 0.01
    p.reset()
    p._ready_gesture_user_on = ready_gesture_user_on  # reset() sets it False; override after
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


class TestReadyGesture(unittest.TestCase):
    """ready_gesture_enabled: while the live ball reading stays inside the SELECTED skill's trained
    box (skill_ball_xy[sel] +- self._skill_ball_halfwidth_xy[sel]), the right arm swings
    CONTINUOUSLY -- amplitude eases in when the ball enters, eases out when it leaves. Locomotion-
    only, standing-only (by default), pure output overlay -- never touches self.last_action or
    anything else."""

    def _p(self, **kw):
        # skill 0 box centered at (1.0, 0.0) +-0.1 ; skill 1 centered at (2.0, 0.5) +-0.2
        kw.setdefault("ready_gesture_user_on", True)  # the [TOGGLE_READY_GESTURE] switch, on by default here
        return _make_policy(
            [0, 100],
            [100, 200],
            {0.0: _identity_quat(), 100.0: _identity_quat()},
            skill_ball_xy=[[1.0, 0.0], [2.0, 0.5]],
            skill_ball_halfwidth_xy=[[0.1, 0.1], [0.2, 0.2]],
            ready_gesture_enabled=True,
            **kw,
        )

    def _stand(self, p):
        p.lin_vel_command = np.zeros(2)
        p.ang_vel_command = 0.0

    def _run(self, p, ball, n):
        """n obs+action ticks with a fixed ball reading; returns the right-arm offset per tick."""
        base = np.zeros(29)
        deltas = []
        for _ in range(n):
            p._update_ready_gesture_state(ball)
            out = p._apply_ready_gesture(base)
            deltas.append(out[[22, 25]].copy())
        return np.array(deltas)

    def test_disabled_by_default_is_a_total_noop(self):
        p = _make_policy([0], [], {0.0: _identity_quat()}, skill_ball_xy=[[1.0, 0.0]])
        self._stand(p)
        d = self._run(p, np.array([1.0, 0.0, 0.11]), 50)
        np.testing.assert_array_equal(d, 0.0)

    def test_box_uses_selected_skills_own_center_and_halfwidth(self):
        p = self._p()
        self.assertEqual(p._selected_skill_ball_box(), ((0.9, 1.1), (-0.1, 0.1)))
        p._selected_skill_id = 1
        self.assertEqual(p._selected_skill_ball_box(), ((1.8, 2.2), (0.3, 0.7)))

    def test_swings_continuously_while_the_ball_stays_in_the_box(self):
        p = self._p()
        self._stand(p)
        d = self._run(p, np.array([1.05, -0.05, 0.11]), 400)  # 8s, ball never leaves
        # keeps oscillating for the whole window (many sign changes, not a one-shot burst)
        sign_changes = np.sum(np.diff(np.sign(d[50:, 0])) != 0)
        self.assertGreater(sign_changes, 10, "should keep swinging, not fire once and stop")
        self.assertGreater(np.abs(d[100:, 0]).max(), 0.3, "shoulder should reach a visible amplitude")
        self.assertTrue(p._ready_gesture_engaged)

    def test_eases_in_then_eases_out_when_the_ball_leaves(self):
        p = self._p()
        self._stand(p)
        d_in = self._run(p, np.array([1.0, 0.0, 0.11]), 200)  # in box -> ramps up + swings
        self.assertGreater(p._ready_gesture_level, 0.99)
        d_out = self._run(p, np.array([5.0, 0.0, 0.11]), 200)  # left box -> eases out to 0
        self.assertEqual(p._ready_gesture_level, 0.0)
        self.assertAlmostEqual(float(np.abs(d_out[-1]).max()), 0.0, places=9)
        # ramp-in wasn't instant: amplitude near t=0 is smaller than at steady state
        self.assertLess(np.abs(d_in[3, 0]), np.abs(d_in[100:, 0]).max())

    def test_ball_in_a_DIFFERENT_skills_box_does_not_engage(self):
        p = self._p()  # selected skill 0, box (0.9,1.1)x(-0.1,0.1)
        self._stand(p)
        self._run(p, np.array([2.0, 0.5, 0.11]), 50)  # skill 1's box, not skill 0's
        self.assertFalse(p._ready_gesture_engaged)
        self.assertEqual(p._ready_gesture_level, 0.0)

    def test_does_not_engage_in_kick_mode(self):
        p = self._p()
        self._stand(p)
        p.task_mode = _TASK_KICK
        self._run(p, np.array([1.0, 0.0, 0.11]), 50)
        self.assertFalse(p._ready_gesture_engaged)

    def test_task_mode_leaving_locomotion_mid_swing_eases_out(self):
        p = self._p()
        self._stand(p)
        self._run(p, np.array([1.0, 0.0, 0.11]), 200)  # engaged, level ~1
        self.assertGreater(p._ready_gesture_level, 0.9)
        p.task_mode = _TASK_KICK
        d = self._run(p, np.array([1.0, 0.0, 0.11]), 100)  # still "in box" but now kicking
        self.assertEqual(p._ready_gesture_level, 0.0)
        self.assertAlmostEqual(float(np.abs(d[-1]).max()), 0.0, places=9)

    def test_does_not_engage_while_walking_when_only_when_standing(self):
        p = self._p()
        p.lin_vel_command = np.array([0.5, 0.0])
        p.ang_vel_command = 0.0
        self._run(p, np.array([1.0, 0.0, 0.11]), 50)
        self.assertFalse(p._ready_gesture_engaged)

    def test_engages_while_walking_if_only_when_standing_is_false(self):
        p = self._p(ready_gesture_only_when_standing=False)
        p.lin_vel_command = np.array([0.5, 0.0])
        p.ang_vel_command = 0.0
        self._run(p, np.array([1.0, 0.0, 0.11]), 50)
        self.assertTrue(p._ready_gesture_engaged)

    def test_none_ball_reading_never_engages(self):
        p = self._p()
        self._stand(p)
        self._run(p, None, 50)
        self.assertFalse(p._ready_gesture_engaged)

    def test_offsets_only_the_right_arm(self):
        p = self._p()
        self._stand(p)
        p._update_ready_gesture_state(np.array([1.0, 0.0, 0.11]))
        for _ in range(30):  # ramp up + get mid-swing
            p._apply_ready_gesture(np.zeros(29))
        base = np.arange(29, dtype=np.float64)
        out = p._apply_ready_gesture(base)
        moved = np.where(~np.isclose(out, base))[0].tolist()
        self.assertEqual(moved, [22, 25], "only right_shoulder_pitch (22) and right_elbow (25) may move")

    def test_noop_when_checkpoint_has_no_skill_ball_xy(self):
        p = _make_policy(
            [0], [], {0.0: _identity_quat()}, ready_gesture_enabled=True, ready_gesture_user_on=True
        )  # toggle ON, but no skill_ball_xy metadata -> still nothing
        self._stand(p)
        self._run(p, np.array([1.0, 0.0, 0.11]), 50)
        self.assertIsNone(p._selected_skill_ball_box())
        self.assertFalse(p._ready_gesture_engaged)


class TestReadyGestureToggle(unittest.TestCase):
    """[TOGGLE_READY_GESTURE]: a runtime ON/OFF master switch (_ready_gesture_user_on). Starts OFF;
    the gesture never engages until it's toggled ON, and eases out when toggled back OFF. Only
    meaningful when ready_gesture_enabled is set in the cfg."""

    def _p(self, **kw):
        return _make_policy(
            [0, 100],
            [100, 200],
            {0.0: _identity_quat(), 100.0: _identity_quat()},
            skill_ball_xy=[[1.0, 0.0], [2.0, 0.5]],
            skill_ball_halfwidth_xy=[[0.1, 0.1], [0.2, 0.2]],
            ready_gesture_enabled=True,
            **kw,
        )

    def test_starts_off_and_reset_clears_it(self):
        p = self._p(ready_gesture_user_on=True)
        self.assertTrue(p._ready_gesture_user_on)
        p.reset()
        self.assertFalse(p._ready_gesture_user_on, "reset() must clear the runtime switch")

    def test_gesture_does_not_engage_while_toggled_off_even_with_ball_in_box(self):
        p = self._p()  # user_on defaults False here
        p.lin_vel_command = np.zeros(2)
        p.ang_vel_command = 0.0
        p._update_ready_gesture_state(np.array([1.0, 0.0, 0.11]))  # ball dead-center in skill 0's box
        self.assertFalse(p._ready_gesture_engaged)

    def test_toggle_command_flips_the_switch(self):
        p = self._p()
        p.post_step_callback(["[TOGGLE_READY_GESTURE]"])
        self.assertTrue(p._ready_gesture_user_on)
        p.post_step_callback(["[TOGGLE_READY_GESTURE]"])
        self.assertFalse(p._ready_gesture_user_on)

    def test_toggling_on_lets_an_in_box_ball_engage_the_gesture(self):
        p = self._p()
        p.lin_vel_command = np.zeros(2)
        p.ang_vel_command = 0.0
        p.post_step_callback(["[TOGGLE_READY_GESTURE]"])  # -> ON
        p._update_ready_gesture_state(np.array([1.0, 0.0, 0.11]))
        self.assertTrue(p._ready_gesture_engaged)

    def test_toggling_off_mid_swing_eases_the_arm_back_out(self):
        p = self._p(ready_gesture_user_on=True)
        p.lin_vel_command = np.zeros(2)
        p.ang_vel_command = 0.0
        for _ in range(200):  # ramp up to full amplitude
            p._update_ready_gesture_state(np.array([1.0, 0.0, 0.11]))
            p._apply_ready_gesture(np.zeros(29))
        self.assertGreater(p._ready_gesture_level, 0.9)
        p.post_step_callback(["[TOGGLE_READY_GESTURE]"])  # -> OFF
        deltas = []
        for _ in range(200):
            p._update_ready_gesture_state(np.array([1.0, 0.0, 0.11]))  # ball still in box, but switch is off
            deltas.append(p._apply_ready_gesture(np.zeros(29))[[22, 25]].copy())
        self.assertEqual(p._ready_gesture_level, 0.0)
        self.assertAlmostEqual(float(np.abs(np.array(deltas)[-1]).max()), 0.0, places=9)

    def test_toggle_command_is_a_warned_noop_when_feature_disabled_in_cfg(self):
        p = _make_policy([0], [], {0.0: _identity_quat()}, ready_gesture_enabled=False)
        p.post_step_callback(["[TOGGLE_READY_GESTURE]"])
        self.assertFalse(p._ready_gesture_user_on, "cfg-disabled -> toggle must not flip anything on")


class TestSkillCycleGesture(unittest.TestCase):
    """skill_cycle_gesture_enabled: [CYCLE_KICK_SKILL] triggers a ONE-SHOT left-arm wave (distinct
    joints/side from the continuous right-arm readiness gesture) that plays for
    skill_cycle_gesture_duration_s and then stops on its own -- no toggle, no ball involved."""

    def _three_skill_policy(self, **kw):
        kw.setdefault("skill_cycle_gesture_enabled", True)
        return _make_policy(
            [0, 100, 300],
            [100, 300, 500],
            {0.0: _identity_quat(), 100.0: _identity_quat(), 300.0: _identity_quat()},
            **kw,
        )

    def _run_action_ticks(self, p, n):
        """n get_action-style ticks (no ONNX call needed -- just the overlay chain) with a fixed
        zero base action; returns the left-arm offset per tick."""
        base = np.zeros(29)
        deltas = []
        for _ in range(n):
            out = p._apply_skill_cycle_gesture(p._apply_ready_gesture(base.copy()))
            deltas.append(out[[15, 18]].copy())
        return np.array(deltas)

    def test_disabled_by_default_is_a_total_noop(self):
        p = self._three_skill_policy(skill_cycle_gesture_enabled=False)
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        d = self._run_action_ticks(p, 60)
        np.testing.assert_array_equal(d, 0.0)

    def test_cycle_press_arms_a_one_shot_wave_that_starts_and_ends_at_zero(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        total_ticks = int(round(p._skill_cycle_gesture_duration_s * p.freq))
        d = self._run_action_ticks(p, total_ticks)
        self.assertAlmostEqual(float(d[0, 0]), 0.0, places=6, msg="must start at exactly zero")
        self.assertAlmostEqual(float(d[-1, 0]), 0.0, places=6, msg="must end at exactly zero")
        self.assertGreater(np.abs(d[:, 0]).max(), 0.1, "shoulder should reach a visible amplitude")

    def test_wave_is_truly_one_shot_and_settles_back_to_zero_and_stays(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        total_ticks = int(round(p._skill_cycle_gesture_duration_s * p.freq))
        self._run_action_ticks(p, total_ticks)  # let the wave fully play out
        d_after = self._run_action_ticks(p, 50)  # no further cycle press
        np.testing.assert_array_equal(d_after, 0.0)
        self.assertEqual(p._skill_cycle_gesture_ticks_left, 0)

    def test_only_the_left_arm_moves_not_the_right(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        base = np.arange(29, dtype=np.float64)
        moved_any = False
        for _ in range(10):
            out = p._apply_skill_cycle_gesture(base.copy())
            moved = np.where(~np.isclose(out, base))[0].tolist()
            if moved:
                moved_any = True
                self.assertEqual(moved, [15, 18], "only left_shoulder_pitch (15) and left_elbow (18) may move")
        self.assertTrue(moved_any)

    def test_repressing_cycle_mid_wave_restarts_it_from_the_top(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        self._run_action_ticks(p, 10)  # partway through the first wave
        ticks_left_before_repress = p._skill_cycle_gesture_ticks_left
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        self.assertGreater(p._skill_cycle_gesture_ticks_left, ticks_left_before_repress)
        self.assertEqual(p._skill_cycle_gesture_ticks_left, p._skill_cycle_gesture_total_ticks)

    def test_does_not_arm_mid_kick_and_kick_in_progress_suppresses_any_armed_wave(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[TRIGGER_KICK:0]"])
        self.assertEqual(p.task_mode, _TASK_KICK)
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])  # pending selection still advances...
        self.assertEqual(p._selected_skill_id, 1)
        self.assertEqual(p._skill_cycle_gesture_ticks_left, 0, "...but no wave arms while kicking")

    def test_task_mode_leaving_locomotion_mid_wave_cuts_it_immediately(self):
        p = self._three_skill_policy()
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        self._run_action_ticks(p, 5)
        self.assertGreater(p._skill_cycle_gesture_ticks_left, 0)
        p.task_mode = _TASK_KICK
        out = p._apply_skill_cycle_gesture(np.zeros(29))
        np.testing.assert_array_equal(out, 0.0)
        self.assertEqual(p._skill_cycle_gesture_ticks_left, 0)

    def test_noop_on_single_skill_checkpoint_cycling_is_itself_a_noop(self):
        p = _make_policy([0], [], {0.0: _identity_quat()}, skill_cycle_gesture_enabled=True)
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        self.assertEqual(p._selected_skill_id, 0)  # nothing to cycle to
        self.assertEqual(p._skill_cycle_gesture_ticks_left, 0, "no wave when there's nothing to cycle")

    def test_independent_of_ready_gesture_both_can_run_at_once_on_opposite_arms(self):
        p = self._three_skill_policy(
            ready_gesture_enabled=True,
            ready_gesture_user_on=True,
            skill_ball_xy=[[1.0, 0.0], [2.0, 0.5], [3.0, 0.0]],
        )
        p.lin_vel_command = np.zeros(2)
        p.ang_vel_command = 0.0
        p._update_ready_gesture_state(np.array([1.0, 0.0, 0.11]))  # right-arm gesture engages
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])  # left-arm wave also arms
        out = p._apply_skill_cycle_gesture(p._apply_ready_gesture(np.zeros(29)))
        # both sides get a chance to move across the run; just confirm neither call raises/zeroes
        # the other out structurally (disjoint indices, both present in the overlay chain).
        self.assertTrue(p._ready_gesture_engaged)
        self.assertGreater(p._skill_cycle_gesture_ticks_left, 0)


class TestManualKickAimTheta(unittest.TestCase):
    """manual_kick_aim_enabled: [KICK_AIM_THETA_INC]/[_DEC]/[_RESET] let the controller dial an
    operator-held kick_aim_theta that REPLACES kick_target_pos_b from the ball-perception controller
    (kick_ball_pos_b is a separate, untouched term)."""

    def _p(self, **kw):
        kw.setdefault("manual_kick_aim_enabled", True)
        return _make_policy(
            [0, 100],
            [100, 200],
            {0.0: _identity_quat(), 100.0: _identity_quat()},
            **kw,
        )

    def test_disabled_by_default_nudge_and_reset_are_noops(self):
        p = _make_policy([0], [], {0.0: _identity_quat()})  # manual_kick_aim_enabled=False default
        p.post_step_callback(["[KICK_AIM_THETA_INC]"])
        self.assertEqual(p._manual_kick_aim_theta_deg, 0.0)
        p._manual_kick_aim_theta_deg = 7.0
        p.post_step_callback(["[KICK_AIM_THETA_RESET]"])
        self.assertEqual(p._manual_kick_aim_theta_deg, 7.0, "reset must also no-op when disabled")

    def test_inc_and_dec_step_by_configured_amount(self):
        p = self._p(manual_kick_aim_step_deg=5.0)
        p.post_step_callback(["[KICK_AIM_THETA_INC]"])
        self.assertAlmostEqual(p._manual_kick_aim_theta_deg, 5.0)
        p.post_step_callback(["[KICK_AIM_THETA_INC]"])
        self.assertAlmostEqual(p._manual_kick_aim_theta_deg, 10.0)
        p.post_step_callback(["[KICK_AIM_THETA_DEC]"])
        self.assertAlmostEqual(p._manual_kick_aim_theta_deg, 5.0)

    def test_reset_zeros_theta(self):
        p = self._p()
        p.post_step_callback(["[KICK_AIM_THETA_INC]", "[KICK_AIM_THETA_INC]"])
        self.assertNotEqual(p._manual_kick_aim_theta_deg, 0.0)
        p.post_step_callback(["[KICK_AIM_THETA_RESET]"])
        self.assertEqual(p._manual_kick_aim_theta_deg, 0.0)

    def test_clamps_to_selected_skills_own_trained_max_deg(self):
        p = self._p(manual_kick_aim_step_deg=10.0, skill_kick_aim_max_deg=[15.0, 15.0])
        for _ in range(5):  # would reach 50 deg unclamped
            p.post_step_callback(["[KICK_AIM_THETA_INC]"])
        self.assertAlmostEqual(p._manual_kick_aim_theta_deg, 15.0)
        for _ in range(10):  # symmetric clamp on the negative side
            p.post_step_callback(["[KICK_AIM_THETA_DEC]"])
        self.assertAlmostEqual(p._manual_kick_aim_theta_deg, -15.0)

    def test_falls_back_to_theta_ref_deg_when_skill_has_no_kick_aim_metadata(self):
        p = self._p(manual_kick_aim_step_deg=100.0, skill_kick_aim_max_deg=[None, None], kick_aim_theta_ref_deg=45.0)
        p.post_step_callback(["[KICK_AIM_THETA_INC]"])
        self.assertAlmostEqual(p._manual_kick_aim_theta_deg, 45.0, "must clamp to theta_ref_deg, not run away")

    def test_uses_the_selected_skills_own_max_not_another_skills(self):
        p = self._p(manual_kick_aim_step_deg=100.0, skill_kick_aim_max_deg=[5.0, 20.0])
        p.post_step_callback(["[KICK_AIM_THETA_INC]"])  # selected skill 0 -> clamp at 5
        self.assertAlmostEqual(p._manual_kick_aim_theta_deg, 5.0)
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])  # now selected skill 1 -> clamp at 20
        p.post_step_callback(["[KICK_AIM_THETA_INC]"])
        self.assertAlmostEqual(p._manual_kick_aim_theta_deg, 20.0)

    def test_target_pos_b_computed_from_theta_over_theta_ref_deg(self):
        p = self._p(skill_kick_aim_max_deg=[45.0, 45.0], kick_aim_theta_ref_deg=45.0)
        p._manual_kick_aim_theta_deg = 15.0
        out = p._manual_kick_aim_target_pos_b()
        np.testing.assert_allclose(out, [15.0 / 45.0, 0.0], atol=1e-6)

    def test_resolve_overrides_target_but_not_ball_pos_when_enabled(self):
        p = self._p()
        ctrl_data = {"BallPoseUdpCtrl": {"valid": True, "kick_ball_pos_b": np.array([1.0, 2.0, 3.0]), "kick_target_pos_b": np.array([9.0, 9.0])}}
        p._manual_kick_aim_theta_deg = 9.0  # -> [0.2, 0.0] at ref_deg=45.0
        ball_pos_b, target_pos_b = p._resolve_ball_and_target(ctrl_data)
        np.testing.assert_array_equal(ball_pos_b, [1.0, 2.0, 3.0])  # untouched -- separate obs term
        np.testing.assert_allclose(target_pos_b, [0.2, 0.0], atol=1e-6)  # overridden, not the ctrl's [9,9]

    def test_resolve_passes_through_ctrl_target_when_disabled(self):
        p = _make_policy([0], [], {0.0: _identity_quat()})  # manual_kick_aim_enabled=False default
        ctrl_data = {"BallPoseUdpCtrl": {"valid": True, "kick_ball_pos_b": np.array([1.0, 2.0, 3.0]), "kick_target_pos_b": np.array([9.0, 9.0])}}
        ball_pos_b, target_pos_b = p._resolve_ball_and_target(ctrl_data)
        np.testing.assert_array_equal(target_pos_b, [9.0, 9.0])  # unchanged legacy behavior

    def test_theta_persists_across_cycle_and_return_to_loco(self):
        p = self._p()
        p.post_step_callback(["[KICK_AIM_THETA_INC]"])
        p.post_step_callback(["[TRIGGER_KICK]"])
        p.post_step_callback(["[RETURN_TO_LOCO]"])
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])
        self.assertAlmostEqual(p._manual_kick_aim_theta_deg, p._manual_kick_aim_step_deg)

    def test_reset_of_the_whole_policy_clears_manual_theta(self):
        p = self._p()
        p.post_step_callback(["[KICK_AIM_THETA_INC]"])
        self.assertNotEqual(p._manual_kick_aim_theta_deg, 0.0)
        p.reset()
        self.assertEqual(p._manual_kick_aim_theta_deg, 0.0)


class TestAutoNav(unittest.TestCase):
    """autonav_enabled: [TOGGLE_AUTONAV] lets the policy compute its own (vx, vy, yaw_rate) toward
    the SELECTED skill's ball box instead of reading w/a/s/d/stick input. Control law:
    speed = min(kp_approach * gap_to_box, max_speed) along unit(ball - box_center); yaw =
    kp_yaw*atan2(nav_error.y, max(ball.x, 0.3)) with a deadband, ONLY while gap > 0.15 m; clamped to
    _cmd_max_mag; zero once inside the box. Cancelled by manual input (checked from RAW ctrl_data,
    before autonav substitution) or a lost/no-metadata ball reading."""

    _MAXV = 0.35  # _make_policy default autonav_max_speed
    _KPA = 1.5  # _make_policy default autonav_kp_approach

    def _p(self, **kw):
        kw.setdefault("autonav_enabled", True)
        kw.setdefault("skill_ball_xy", [[1.0, 0.0], [2.0, 0.5]])
        kw.setdefault("skill_ball_halfwidth_xy", [[0.1, 0.1], [0.2, 0.2]])
        return _make_policy(
            [0, 100],
            [100, 200],
            {0.0: _identity_quat(), 100.0: _identity_quat()},
            **kw,
        )

    # ---- [TOGGLE_AUTONAV] command ----

    def test_toggle_is_a_warned_noop_when_disabled_in_cfg(self):
        p = self._p(autonav_enabled=False)
        p.post_step_callback(["[TOGGLE_AUTONAV]"])
        self.assertFalse(p._autonav_user_on)

    def test_toggle_flips_the_switch(self):
        p = self._p()
        p.post_step_callback(["[TOGGLE_AUTONAV]"])
        self.assertTrue(p._autonav_user_on)
        p.post_step_callback(["[TOGGLE_AUTONAV]"])
        self.assertFalse(p._autonav_user_on)

    def test_reset_clears_the_switch(self):
        p = self._p()
        p.post_step_callback(["[TOGGLE_AUTONAV]"])
        self.assertTrue(p._autonav_user_on)
        p.reset()
        self.assertFalse(p._autonav_user_on)

    # ---- _compute_autonav_cmd control law ----

    def test_holds_zero_once_ball_is_inside_the_box(self):
        p = self._p()  # skill 0 box: (0.9,1.1) x (-0.1,0.1)
        cmd = p._compute_autonav_cmd(np.array([1.0, 0.0, 0.11]))
        np.testing.assert_array_equal(cmd, [0.0, 0.0, 0.0])

    def test_drives_toward_the_box_center_with_the_right_sign(self):
        p = self._p()  # skill 0 box centered (1.0, 0.0); ball farther forward AND to the left
        ball = np.array([1.5, 0.2, 0.11])
        cmd = p._compute_autonav_cmd(ball)
        # nav_error = (0.5, 0.2) -> vx, vy both positive (walk forward + left); yaw positive (left)
        self.assertGreater(cmd[0], 0.0)
        self.assertGreater(cmd[1], 0.0)
        self.assertGreater(cmd[2], 0.0, "ball to the left -> turn left (positive yaw)")
        # command points along unit(nav_error): vy/vx == nav_error.y/nav_error.x
        self.assertAlmostEqual(cmd[1] / cmd[0], 0.2 / 0.5, places=5)

    def test_opposite_error_gives_opposite_sign_command(self):
        p = self._p()
        cmd = p._compute_autonav_cmd(np.array([1.5, -0.2, 0.11]))  # forward AND to the right
        self.assertGreater(cmd[0], 0.0)
        self.assertLess(cmd[1], 0.0, "ball to the right of target -> walk right -> negative vy")
        self.assertLess(cmd[2], 0.0, "ball to the right -> turn right (negative yaw)")

    def test_speed_is_capped_at_max_speed_when_far_from_the_box(self):
        p = self._p()  # box (0.9,1.1)x(-0.1,0.1)
        cmd = p._compute_autonav_cmd(np.array([4.0, 0.0, 0.11]))  # ball 2.9 m past the far edge
        self.assertAlmostEqual(float(np.linalg.norm(cmd[:2])), self._MAXV, places=6)

    def test_speed_scales_DOWN_with_the_box_gap_near_the_zone(self):
        # THE FIX: closing speed is proportional to the distance to the box BOUNDARY, not the
        # centre -- so the robot is already crawling by the time the ball reaches the zone instead
        # of blowing through it. Ball just 5 cm past the near edge -> tiny speed, not kp*halfwidth.
        p = self._p()  # box x in (0.9, 1.1)
        cmd = p._compute_autonav_cmd(np.array([1.15, 0.0, 0.11]))  # gap_x = 1.15 - 1.10 = 0.05
        speed = float(np.linalg.norm(cmd[:2]))
        self.assertAlmostEqual(speed, self._KPA * 0.05, places=5)
        self.assertLess(speed, 0.1, "a 5 cm gap must command a crawl, not a stride")

    def test_yaw_is_dropped_entirely_once_within_the_min_gap_of_the_box(self):
        p = self._p()  # box (0.9,1.1)x(-0.1,0.1), _AUTONAV_YAW_MIN_GAP_M = 0.15
        # ball 5 cm past the near edge and 12 cm off-axis: still outside, but gap ~0.13 < 0.15
        cmd = p._compute_autonav_cmd(np.array([1.15, 0.22, 0.11]))
        self.assertEqual(cmd[2], 0.0, "near the zone, vx/vy do the fine positioning -- no residual turn")
        self.assertGreater(np.linalg.norm(cmd[:2]), 0.0, "but it's still nudging into the box")

    def test_yaw_applies_and_is_well_conditioned_while_still_far(self):
        p = self._p()  # ball far and to the left -> real turn wanted, and it must not saturate
        cmd = p._compute_autonav_cmd(np.array([2.5, 0.3, 0.11]))
        self.assertGreater(cmd[2], 0.0)
        # denominator is max(ball.x, 0.3) = 2.5, NOT nav_error.x -- a few-deg turn, not a spin
        np.testing.assert_allclose(cmd[2], 2.0 * np.arctan2(0.3, 2.5), atol=1e-6)

    def test_yaw_deadband_zeros_sub_degree_heading_residuals(self):
        p = self._p()
        # ball far in x (gap > 0.15 so yaw is eligible) but essentially dead-ahead laterally
        cmd = p._compute_autonav_cmd(np.array([2.0, 0.005, 0.11]))
        self.assertGreater(cmd[0], 0.0, "still needs to walk forward")
        self.assertEqual(cmd[2], 0.0, "a <4deg heading error must not make the heading hunt")

    def test_yaw_denominator_floored_when_ball_far_but_nearly_beside_the_robot(self):
        p = self._p()
        cmd = p._compute_autonav_cmd(np.array([0.2, 1.5, 0.11]))  # ball way out to the left, close in x
        # fwd_for_yaw = max(0.2, 0.3) = 0.3 (floored) -> atan2(1.5, 0.3), bounded, clamps to a real turn
        np.testing.assert_allclose(cmd[2], np.clip(2.0 * np.arctan2(1.5, 0.3), -0.8, 0.8), atol=1e-6)

    def test_linear_command_clamps_to_cmd_max_mag(self):
        p = self._p(autonav_kp_approach=100.0, autonav_max_speed=100.0)
        cmd = p._compute_autonav_cmd(np.array([5.0, 5.0, 0.11]))
        self.assertLessEqual(abs(cmd[0]), p._cmd_max_mag[0] + 1e-9)
        self.assertLessEqual(abs(cmd[1]), p._cmd_max_mag[1] + 1e-9)

    def test_cancels_and_freezes_when_ball_reading_is_none(self):
        p = self._p()
        p._autonav_user_on = True
        cmd = p._compute_autonav_cmd(None)
        np.testing.assert_array_equal(cmd, [0.0, 0.0, 0.0])
        self.assertFalse(p._autonav_user_on)

    def test_cancels_when_selected_skill_has_no_box_metadata(self):
        p = _make_policy([0], [], {0.0: _identity_quat()}, autonav_enabled=True)  # no skill_ball_xy
        p._autonav_user_on = True
        cmd = p._compute_autonav_cmd(np.array([1.0, 0.0, 0.11]))
        np.testing.assert_array_equal(cmd, [0.0, 0.0, 0.0])
        self.assertFalse(p._autonav_user_on)

    def test_zero_mid_kick_without_cancelling_the_switch(self):
        p = self._p()
        p._autonav_user_on = True
        p.task_mode = _TASK_KICK
        cmd = p._compute_autonav_cmd(np.array([1.5, 0.2, 0.11]))
        np.testing.assert_array_equal(cmd, [0.0, 0.0, 0.0])
        self.assertTrue(p._autonav_user_on, "mid-kick is transient -- must resume in locomotion, not need re-toggling")

    def test_retargets_live_to_the_newly_selected_skill_after_cycling(self):
        p = self._p()  # skill 0 box centered (1.0,0.0), skill 1 centered (2.0,0.5)
        p.post_step_callback(["[CYCLE_KICK_SKILL]"])  # -> selected skill 1
        ball = np.array([2.0, 0.5, 0.11])  # dead-center of skill 1's box
        cmd = p._compute_autonav_cmd(ball)
        np.testing.assert_array_equal(cmd, [0.0, 0.0, 0.0], "already at skill 1's target -- must hold, not skill 0's")

    # ---- _update_velocity_command dispatch + manual-override cancellation ----

    def test_manual_keyboard_input_cancels_engaged_autonav(self):
        p = self._p()
        p._autonav_user_on = True
        ctrl_data = {"KeyboardCtrl": {"keyboard_event": [{"type": "keyboard", "name": "w", "pressed": True}]}}
        p._update_velocity_command(ctrl_data, np.array([1.5, 0.2, 0.11]))
        self.assertFalse(p._autonav_user_on)
        self.assertGreater(p.lin_vel_command[0], 0.0, "manual forward command must win, not autonav's")

    def test_manual_joystick_input_beyond_deadzone_cancels_engaged_autonav(self):
        p = self._p()
        p._autonav_user_on = True
        ctrl_data = {"JoystickCtrl": {"axes": {"LeftX": 0.0, "LeftY": 0.5, "RightX": 0.0}}}
        p._update_velocity_command(ctrl_data, np.array([1.5, 0.2, 0.11]))
        self.assertFalse(p._autonav_user_on)

    def test_joystick_noise_within_deadzone_does_not_cancel_autonav(self):
        p = self._p(autonav_manual_deadzone=0.05)
        p._autonav_user_on = True
        ctrl_data = {"JoystickCtrl": {"axes": {"LeftX": 0.01, "LeftY": 0.02, "RightX": 0.0}}}
        p._update_velocity_command(ctrl_data, np.array([1.5, 0.2, 0.11]))
        self.assertTrue(p._autonav_user_on)

    def test_no_manual_input_dispatches_to_autonav_when_engaged(self):
        p = self._p()
        p._autonav_user_on = True
        ctrl_data = {"KeyboardCtrl": {"keyboard_event": []}}
        p._update_velocity_command(ctrl_data, np.array([1.5, 0.2, 0.11]))
        self.assertTrue(p._autonav_user_on)
        self.assertGreater(p.lin_vel_command[0], 0.0)
        self.assertGreater(p.lin_vel_command[1], 0.0)

    def test_manual_input_flows_through_unaffected_when_autonav_off(self):
        p = self._p()  # autonav_user_on starts False (reset() default)
        ctrl_data = {"KeyboardCtrl": {"keyboard_event": [{"type": "keyboard", "name": "w", "pressed": True}]}}
        p._update_velocity_command(ctrl_data, None)
        self.assertGreater(p.lin_vel_command[0], 0.0)


if __name__ == "__main__":
    unittest.main()
