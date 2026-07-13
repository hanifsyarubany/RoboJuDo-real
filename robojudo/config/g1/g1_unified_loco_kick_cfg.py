"""Deploy the holosoma unified locomotion + ball-kicking G1 policy in RoboJuDo.

Two registered configs, run with ``python scripts/run_pipeline_prepared.py -c <name>`` (NOT
``scripts/run_pipeline.py`` -- see the note below):
  - ``g1_unified_loco_kick``      : the policy alone (sim2sim or real via DEPLOY_TARGET below).
  - ``g1_unified_loco_kick_amo``  : G1 AMO as the INITIAL policy for driving the robot around,
                                    then switch to our policy at runtime (multi-policy pipeline).

Use ``run_pipeline_prepared.py``, not the stock ``run_pipeline.py``: the stock script only runs
the ramp-to-default-pose sequence (``pipeline.prepare()``) on real hardware; in sim it silently
skips straight to the control loop from MuJoCo's raw spawn pose (q=0 for every joint -- holosoma's
model has no <keyframe> of its own), which corrupts every dof_pos-relative observation from tick
zero. In sim, the prepared launcher resets to a grounded ``default_stand`` keyframe (added to
``scene_g1_29dof.xml``) and hands control to the policy immediately -- NOT via ``prepare()``'s
open-loop PD ramp, which was verified to topple the robot on its own even from a correct starting
pose (this pose is only *actively* stable, i.e. it needs the trained policy's continuous balance
corrections from frame 0, same as holosoma's own reference sim2sim setup, which supports the robot
on a gantry until the policy is already running). On real hardware, ``prepare()`` still runs as
before -- the robot there starts from whatever pose a technician left it in, not a known spawn.

===================================  SWITCHES  ===================================
Everything you'd flip between a sim2sim test and a real-robot deploy is a single knob here:

  DEPLOY_TARGET = "sim"    ->  MuJoCo sim2sim (G1MujocoEnvCfg)
                = "real"   ->  real Unitree G1 via unitree_cpp (G1RealEnvCfg, env_type=UnitreeCppEnv)

  CONTROLLER    = "both"   ->  keyboard AND joystick both active (use whichever; default)
                = "keyboard"
                = "joystick"
  (On real hardware the robot's own controller (UnitreeCtrl) is always available; CONTROLLER just
   adds keyboard on top if you have one connected. If no DISPLAY is set -- e.g. a headless SSH
   session onboard the robot -- keyboard is skipped automatically with a warning instead of
   crashing (pynput's keyboard backend needs an X server); CONTROLLER="keyboard" explicitly in
   that situation is a hard error instead, since there's no other controller to fall back to.)

  NET_IF        = network interface to the robot (only used when DEPLOY_TARGET="real").
=================================================================================

CONTROLS
  Locomotion (drive the robot):
    keyboard : w/s forward/back, a/d strafe, q/e turn        joystick : left stick move, right stick turn
  Kick:
    keyboard : k = trigger kick, l = return to locomotion    joystick : RB+Up = kick, RB+Down = return
  (the kick auto-returns to locomotion when the clip finishes; l / RB+Down is a manual override)
  Stop: keyboard Esc / joystick A (emergency stop on real).
"""

import logging
import os

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import JoystickCtrlCfg, KeyboardCtrlCfg, UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlMultiPolicyPipelineCfg, RlPipelineCfg

logger = logging.getLogger(__name__)

from .env.g1_holosoma_env_cfg import G1HolosomaMujocoEnvCfg, G1HolosomaRealEnvCfg
from .policy.g1_amo_policy_cfg import G1AmoPolicyCfg
from .policy.g1_unified_loco_kick_policy_cfg import G1UnifiedLocoKickPolicyCfg

# ============================== SWITCHES (edit me) ============================== #
DEPLOY_TARGET = "sim"  # "sim" | "real"
CONTROLLER = "both"  # "both" | "keyboard" | "joystick"
NET_IF = "eth0"  # robot network interface (only for DEPLOY_TARGET="real")
# =============================================================================== #

_KB_KICK_TRIGGERS = {"k": "[TRIGGER_KICK]", "l": "[RETURN_TO_LOCO]"}
_JS_KICK_TRIGGERS = {"RB+Up": "[TRIGGER_KICK]", "RB+Down": "[RETURN_TO_LOCO]"}

# The real Unitree remote (UnitreeCtrl, via unitreeRemoteController.button_map) names its shoulder
# buttons "L1"/"R1"; a generic gamepad (JoystickCtrl, via JoystickThread's Xbox-style button_map)
# names the same physical buttons "LB"/"RB". UnitreeCtrlCfg.combination_init_buttons=["L1","R1"]
# means a real "RB held + D-pad Up" press computes the combo key "R1+Up", not "RB+Up" -- so trigger
# dicts written with "RB"/"LB" (matching the docs/guide, which describe the generic-gamepad layout)
# silently never fire on real hardware unless remapped. Keep triggers written in "RB"/"LB" terms
# everywhere and remap only when building the real controller.
_UNITREE_BUTTON_ALIASES = {"LB": "L1", "RB": "R1"}


def _remap_combo_keys(triggers: dict[str, str], aliases: dict[str, str]) -> dict[str, str]:
    """Rewrite 'RB+Up'-style trigger keys' button names via aliases (e.g. RB -> R1)."""
    remapped = {}
    for key, cmd in triggers.items():
        parts = [aliases.get(p, p) for p in key.split("+")]
        remapped["+".join(parts)] = cmd
    return remapped


def _make_env():
    """sim2sim vs real onboard, from the single DEPLOY_TARGET switch.

    Both use the *holosoma-trained* G1 model (see g1_holosoma_env_cfg.py): in sim this makes the
    MuJoCo dynamics match what the policy was trained/tuned against (closing the sim2sim gap); on
    real hardware the dynamics are the physical robot, but FK (torso_quat) stays consistent with
    training. NET_IF is applied to the real config."""
    if DEPLOY_TARGET == "real":
        cfg = G1HolosomaRealEnvCfg()
        cfg.unitree.net_if = NET_IF
        return cfg
    if DEPLOY_TARGET == "sim":
        return G1HolosomaMujocoEnvCfg()
    raise ValueError(f"DEPLOY_TARGET must be 'sim' or 'real', got {DEPLOY_TARGET!r}")


def _make_ctrl(policy_switch_triggers: dict | None = None):
    """Keyboard and/or joystick, plus the robot controller on real hardware. policy_switch_triggers
    (multi-policy configs only) are merged in so the same devices also switch policies."""
    kb_extra = dict(policy_switch_triggers or {})
    js_extra = dict(policy_switch_triggers or {})
    ctrls = []
    want_kb = CONTROLLER in ("both", "keyboard")
    want_js = CONTROLLER in ("both", "joystick")

    # KeyboardCtrl imports pynput at module-import time, which on Linux needs an X server -- fails
    # hard (crashes pipeline construction) when run headless, e.g. onboard the robot's own compute
    # over SSH with no DISPLAY. CONTROLLER="both" is a sane default on a workstation with a screen
    # but not onboard, so degrade gracefully there instead of making the switch a manual chore.
    if want_kb and not os.environ.get("DISPLAY"):
        if CONTROLLER == "keyboard":
            raise RuntimeError(
                "CONTROLLER='keyboard' but no DISPLAY is set -- pynput's keyboard backend needs an "
                "X server, so this won't work over a headless SSH session (e.g. onboard the "
                "robot). Set CONTROLLER='joystick' here, or run from a session with a display."
            )
        logger.warning(
            "No DISPLAY set -- skipping the keyboard controller (pynput needs an X server); "
            "keeping joystick/UnitreeCtrl only. This is expected when running headless onboard "
            "the robot. Set CONTROLLER='joystick' explicitly to silence this."
        )
        want_kb = False

    if want_kb:
        ctrls.append(
            KeyboardCtrlCfg(
                triggers={"Key.esc": "[SHUTDOWN]", **_KB_KICK_TRIGGERS},
                triggers_extra=kb_extra,
            )
        )
    if DEPLOY_TARGET == "real":
        # the robot's own controller is always available on hardware (emergency stop = A). Remap
        # RB/LB -> R1/L1 (see _UNITREE_BUTTON_ALIASES) so combo triggers actually fire.
        real_triggers = _remap_combo_keys({**_JS_KICK_TRIGGERS, **js_extra}, _UNITREE_BUTTON_ALIASES)
        ctrls.append(UnitreeCtrlCfg(triggers_extra=real_triggers))
    elif want_js:
        ctrls.append(JoystickCtrlCfg(triggers_extra={**_JS_KICK_TRIGGERS, **js_extra}))

    if not ctrls:
        raise ValueError(f"no controller selected (CONTROLLER={CONTROLLER!r}, DEPLOY_TARGET={DEPLOY_TARGET!r})")
    return ctrls


@cfg_registry.register
class g1_unified_loco_kick(RlPipelineCfg):
    """Unified locomotion + ball-kicking G1 policy, alone. sim2sim or real via DEPLOY_TARGET."""

    robot: str = "g1"
    env: object = _make_env()
    ctrl: list = _make_ctrl()
    policy: G1UnifiedLocoKickPolicyCfg = G1UnifiedLocoKickPolicyCfg()

    do_safety_check: bool = DEPLOY_TARGET == "real"


@cfg_registry.register
class g1_unified_loco_kick_amo(RlMultiPolicyPipelineCfg):
    """G1 AMO as the INITIAL policy (drive the robot around with the controller), then switch to the
    unified loco+kick policy at runtime.

    Switch policies:  keyboard  [ = AMO (0), ] = unified loco+kick (1)   (see triggers below)
                      joystick  RB+Left = AMO (0),  RB+Right = unified loco+kick (1)
    The kick trigger keys (k / RB+Up) still apply once you're on the unified policy.
    """

    robot: str = "g1"
    env: object = _make_env()
    # policies[0] is the startup policy -> AMO first, then switch to ours (index 1).
    ctrl: list = _make_ctrl(
        policy_switch_triggers={
            # keyboard: KeyboardCtrl.process_triggers has no combination-key logic (that's
            # JoystickCtrl-only) -- these fire on plain [ / ] press+release, no Ctrl needed.
            "[": "[POLICY_SWITCH],0",
            "]": "[POLICY_SWITCH],1",
            # joystick: with RB held -> Left = AMO, Right = ours
            "RB+Left": "[POLICY_SWITCH],0",
            "RB+Right": "[POLICY_SWITCH],1",
        }
    )

    policies: list = [
        G1AmoPolicyCfg(),
        G1UnifiedLocoKickPolicyCfg(),
    ]

    do_safety_check: bool = DEPLOY_TARGET == "real"
