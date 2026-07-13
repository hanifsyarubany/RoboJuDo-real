<h1 align="center"><b>G1 Unified Locomotion + Kick — Deployment Guide</b></h1>

This covers running the holosoma unified locomotion + ball-kicking G1 policy in RoboJuDo: sim2sim
(MuJoCo, on your workstation) and real deployment (onboard the G1). Config lives in
[`robojudo/config/g1/g1_unified_loco_kick_cfg.py`](../robojudo/config/g1/g1_unified_loco_kick_cfg.py).

For generic Unitree SDK installation (`unitree_sdk2py` / `unitree_cpp`) and network setup, see
[`unitree_setup.md`](unitree_setup.md) first if you haven't done that yet — this guide assumes it's
already done for real deployment.

---

# 🚀 Quick Start

Always use `scripts/run_pipeline_prepared.py`, **not** the stock `scripts/run_pipeline.py`. The
stock script only runs the pose-ramp sequence (`pipeline.prepare()`) on real hardware; in sim it
skips straight into the control loop from MuJoCo's raw spawn pose, which corrupts the policy's
observations from tick zero. `run_pipeline_prepared.py` fixes this for sim (resets to a grounded
default-stand pose first) and preserves the correct real-hardware ramp behavior — see the module
docstring in that script for the full "why."

```bash
conda activate robojudo
cd /path/to/RoboJuDo

# sim2sim (MuJoCo, on your workstation)
python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick

# same, but with G1 AMO driving first, switchable to our policy at runtime
python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick_amo

# dry run first (recommended, especially before real hardware) -- see "Dry Run" section below
python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick --dry-run
```

There are only two registered configs:

| Config | What it does |
|---|---|
| `g1_unified_loco_kick` | The policy alone. |
| `g1_unified_loco_kick_amo` | G1 AMO drives the robot first; switch to our policy at runtime. |

---

# ⚙️ The Three Switches

Everything you'd change between a sim2sim test and a real deploy is at the top of
[`g1_unified_loco_kick_cfg.py`](../robojudo/config/g1/g1_unified_loco_kick_cfg.py):

```python
DEPLOY_TARGET = "sim"   # "sim" | "real"
CONTROLLER = "both"     # "both" | "keyboard" | "joystick"
NET_IF = "eth0"          # robot network interface (only used when DEPLOY_TARGET="real")
```

| Switch | Value | Effect |
|---|---|---|
| `DEPLOY_TARGET` | `"sim"` | MuJoCo sim2sim, on the holosoma-calibrated G1 model (`G1HolosomaMujocoEnvCfg`). |
| | `"real"` | Real G1 via `UnitreeCppEnv` (`G1HolosomaRealEnvCfg`). |
| `CONTROLLER` | `"both"` | Keyboard **and** joystick both active — use whichever's convenient. |
| | `"keyboard"` | Keyboard only. |
| | `"joystick"` | A generic gamepad only (see [Joystick](#-joystick-controls) below — **not** the robot's own Unitree remote; see that section for why). |
| `NET_IF` | e.g. `"eth0"` | Network interface to the robot. Only read when `DEPLOY_TARGET="real"`. |

Edit the file, save, and re-run — no other code changes needed to switch between sim2sim and real.

On real hardware, the robot's own wireless remote (`UnitreeCtrl`) is *always* available regardless
of `CONTROLLER` — that switch only controls whether keyboard/generic-gamepad input is *also* wired
in on top of it.

---

# 🖥️ Sim2Sim (MuJoCo, on your workstation)

1. Set `DEPLOY_TARGET = "sim"` in the config (this is the default).
2. Run:
   ```bash
   conda activate robojudo
   python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick
   ```
3. A MuJoCo viewer window opens with the robot already standing (reset to a grounded default pose
   — no gantry/lowering step needed, unlike holosoma's own `run_sim.py` workflow). The policy is
   controlling it immediately.
4. Use keyboard and/or joystick as described below to walk it around and trigger kicks.

No ONNX path setup needed for the default checkpoint — it's baked into
[`g1_unified_loco_kick_policy_cfg.py`](../robojudo/config/g1/policy/g1_unified_loco_kick_policy_cfg.py)'s
`DEFAULT_ONNX_PATH`. To test a different checkpoint, edit that file (or override
`onnx_path` on `G1UnifiedLocoKickPolicyCfg` directly).

---

# 🤖 Real Deployment (onboard G1)

**Do a [dry run](#-dry-run-test-without-moving-the-robot) first.** Especially the first time you
run a given checkpoint on real hardware, or after any change to the policy/config.

1. Complete Unitree SDK setup — see [`unitree_setup.md`](unitree_setup.md).
2. Set `DEPLOY_TARGET = "real"` and `NET_IF` to your robot's network interface.
3. Position the robot safely (enough clearance to walk, ideally supported/spotted for the first
   run) and make sure the emergency-stop path is understood by whoever's operating it (see
   [Controls](#-keyboard-controls) below — **A** on the joystick / **Esc** on the keyboard is
   `[SHUTDOWN]`).
4. Run:
   ```bash
   conda activate robojudo
   python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick
   ```
   This calls `pipeline.prepare()` on real hardware: a 3-second smooth ramp from the robot's
   current joint angles to the trained default pose, then a 5-second blend into policy control.
   Watch the robot during this ramp — it's the same mechanism used for every other RoboJuDo policy
   on real G1.
5. Once `prepare()` finishes, the policy is live. Use the robot's own wireless remote
   (`UnitreeCtrl`) — see [Joystick Controls](#-joystick-controls) below, same button map.

---

# 🧪 Dry Run: test without moving the robot

Pass `--dry-run` to compute observations and actions every tick — with keyboard/joystick commands
fully live — **without ever applying torque**. Nothing physically moves (sim or real). Use this to
review a checkpoint's commanded actions before trusting it to actually move, especially before a
real-hardware run.

```bash
python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick --dry-run
```

What it does differently from a normal run:
- `env.step()` is skipped every tick (same mechanism `self_check()` already uses internally for
  its 10-step startup warmup) — the robot/sim stays frozen in whatever pose it started in.
- On real hardware, `pipeline.prepare()`'s pose-ramp is **also** skipped (it moves the robot via
  real torque, independent of the dry-run flag otherwise) — actions are evaluated against the
  robot's current as-found pose instead.
- Every tick's commanded action is checked for unsafe conditions and logged as a `WARNING` if any
  of these trip:

  | Check | What it means |
  |---|---|
  | NaN/Inf in the action | Something in the observation or network output broke. |
  | Raw action saturating `action_clip` (±100) | The network wants to output more than it's allowed to — a sign of erratic/unstable behavior. |
  | Commanded joint angle outside the model's joint range | Physically impossible / would hit a hard limit. |
  | Would-be torque beyond the joint's torque limit | The PD target implies more torque than the motor can deliver — would be clipped in a real run, degrading tracking. |
  | Large tick-to-tick action jump (>5 rad) | Jerky/unstable output between consecutive ticks. |

  An `INFO` line is also printed every 50 ticks with the current task mode, commanded velocity, and
  max joint-position error, so you can see the log is actually progressing even when nothing trips
  a warning.

If a dry run comes back clean (no warnings, sensible-looking commanded positions in the periodic
`INFO` lines), that's a reasonable pre-flight check — it is **not** a guarantee the robot will
balance/walk correctly once it's actually moving (dry run can't see anything that only emerges from
real dynamics, like falling). Follow it with a real (but careful, supported) run.

---

# ⌨️ Keyboard Controls

Works identically in sim and real (when `CONTROLLER` includes `"keyboard"`).

| Keys | Effect |
|---|---|
| `w` / `s` | forward / back |
| `a` / `d` | strafe left / right |
| `q` / `e` | turn left / right |
| `k` | trigger kick |
| `l` | return to locomotion (manual override — the kick also auto-returns once the clip finishes) |
| `Esc` | emergency stop (`[SHUTDOWN]`) |

**`g1_unified_loco_kick_amo` only** — switch which policy is active:

| Key | Effect |
|---|---|
| `[` | switch to AMO |
| `]` | switch to our unified loco+kick policy |

(No modifier key needed — just press and release `[` or `]`. The kick keys above still work once
you're on the unified policy.)

Movement keys are **held**, not tapped — holding `w` ramps the commanded forward velocity up
smoothly (~0.5s to reach max) and back down smoothly on release, rather than snapping instantly.

---

# 🎮 Joystick Controls

**Sim2sim:** plug in a **generic** USB/Bluetooth gamepad (standard Xbox-style layout: A/B/X/Y,
LB/RB, two sticks) — this is read via SDL/pygame (`JoystickCtrl`) directly on your workstation, no
robot needed.

**Real hardware:** the robot's own Unitree wireless remote is used automatically (`UnitreeCtrl`) —
same button names/layout below, just physically the robot's remote instead of a PC gamepad.

> **Why you can't use the real Unitree remote for sim2sim:** the remote's data is relayed *through
> the robot's own onboard software* (`UnitreeCppEnv`) — `MujocoEnv` has no equivalent relay, so
> there's currently no path for the physical remote to drive a sim2sim run. Use a generic gamepad
> for sim2sim instead.

Layout reference (RB = right shoulder button, above RT; LB = left shoulder, above LT):

```
        [LB]                    [RB]
        [LT]                    [RT]
  ┌─────────────────────────────────┐
  │   ↑                      Y      │
  │ ← D →      (sticks)    X   B    │
  │   ↓                      A      │
  └─────────────────────────────────┘
```

**Movement:**

| Input | Effect |
|---|---|
| Left stick Y | forward / back |
| Left stick X | strafe left / right |
| Right stick X | turn (yaw) |

**Kick** (hold RB, then tap the D-pad):

| Combo | Effect |
|---|---|
| RB + D-pad Up | trigger kick |
| RB + D-pad Down | return to locomotion (manual override) |

**Other:**

| Button | Effect |
|---|---|
| A | emergency stop (`[SHUTDOWN]`) |

**`g1_unified_loco_kick_amo` only:**

| Combo | Effect |
|---|---|
| RB + D-pad Left | switch to AMO |
| RB + D-pad Right | switch to our unified loco+kick policy |

Like the keyboard, stick deflection is rate-limited into the commanded velocity (~0.5s to reach max
magnitude) rather than applied instantly.

---

# 🩺 Troubleshooting

- **`No joystick connected` error in the log** — no gamepad was detected by SDL at startup. Either
  plug one in before launching, or set `CONTROLLER = "keyboard"` to silence it (keyboard still works
  fine on its own).
- **`sim env has no 'default_stand' keyframe` error** — you're pointing `DEPLOY_TARGET="sim"` at an
  env config whose XML doesn't have the `default_stand` keyframe (only
  `assets/robots/g1/holosoma_model/scene_g1_29dof.xml` has it). Don't swap in a different XML
  without adding an equivalent keyframe, or expect the pre-run spawn-pose fix to not apply.
- **Frame drops / `Warning: frame drop` spam on real hardware** — usually a compute or network
  bottleneck; the script exits automatically if drops exceed -0.2s to avoid degraded control.
- **Robot falls immediately in sim** — check you're on `run_pipeline_prepared.py`, not
  `run_pipeline.py`; the latter has the known spawn-pose bug for this policy in sim.
- **Checkpoint feels wobbly/jittery on real hardware** — run a `--dry-run` first and check the
  warning log for saturating actions or torque-limit violations before assuming it's a tuning
  issue with the deploy stack itself.
