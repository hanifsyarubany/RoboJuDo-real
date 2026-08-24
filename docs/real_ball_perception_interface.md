<h1 align="center"><b>Real Ball Perception — Interface Spec &amp; Setup Guide</b></h1>

This is the spec for whoever builds the real onboard ball detector for the G1: exactly what to
publish, in what format, at what rate, plus the step-by-step to get it feeding
`g1_unified_loco_kick` on real hardware. Read
[`g1_unified_loco_kick_deployment.md`](g1_unified_loco_kick_deployment.md) first for the general
real-deployment flow — this doc is specifically about the perception side of `--live-ball`.

---

# 🗺️ Architecture — why it has this shape

```
 REAL DETECTOR              BRIDGE                        ROBOJUDO
 (your code, new)     foxy_ros2_ball_bridge.py       (already built, unmodified)
 ┌──────────────┐     ┌──────────────────────┐       ┌───────────────────────────┐
 │  camera /    │ ROS2│  subscribes to the    │ Redis │ run_pipeline_prepared.py  │
 │  lidar, etc. │────▶│  same topics, writes  │──────▶│  --live-ball              │
 │              │     │  to redis             │       │  --ball-source redis      │
 └──────────────┘     └──────────────────────┘       └───────────────────────────┘
   Foxy, native          Foxy, native                   Humble, py3.12
   (robot's own ROS2)    (same distro as detector,       (needs py3.12 for
                          zero cross-version risk)        onnxruntime+rclpy
                                                           to coexist)
```

**Why a bridge instead of your detector talking to `robojudo` directly over ROS2:** `robojudo`
needs Python 3.12 — the only Humble `rclpy` build that coexists with `onnxruntime` (every
Humble/py3.11 build forces `numpy<2`, which breaks `onnxruntime` outright). The G1's onboard ROS2
is Foxy (Python 3.8). Whether a Humble node and a Foxy node actually interoperate over the DDS wire
is **unverified** — attempting to reconstruct Foxy's `rclpy` to test it hit real, unfixable C++ ABI
breaks in RoboStack's (unmaintained) Foxy packaging. Rather than bet the robot on that, the bridge
keeps ROS2 entirely inside Foxy (talking to itself — zero risk) and only crosses the Humble/Foxy
boundary over Redis, a transport that's already verified working.

**You only need to build the "REAL DETECTOR" box.** The bridge (`scripts/foxy_ros2_ball_bridge.py`)
and everything on the `robojudo` side already exist and are already tested.

---

# 📡 Data Specification

Two ROS2 topics, both published **by your detector**, both consumed **by the bridge**. This is the
exact same contract `robojudo`'s own `BallPoseRos2Ctrl` uses, so if Humble↔Foxy compatibility is
ever confirmed later, your detector plugs in directly with zero changes.

## `/ball_pose` — required

| | |
|---|---|
| **Type** | `geometry_msgs/PointStamped` |
| **Rate** | ~30 Hz (matches `dummy_ball_perception.py`'s default; the control loop itself runs at 50 Hz, so don't publish much slower than 30 Hz or the reading will frequently read as stale) |
| **QoS** | Any — the bridge subscribes `BEST_EFFORT`/`VOLATILE`/depth 5, the most permissive a subscriber can offer, so it connects whether you publish `BEST_EFFORT` or `RELIABLE` |
| **Frame** | Robot **heading frame**: origin at the robot's base, **x = forward, y = left, z = up** (standard right-handed, ROS REP-103 body-frame convention) — see [Coordinate frame](#-coordinate-frame-details) below for exactly how to compute this |
| **Units** | Meters |

Fields used:

```
header.stamp     -- detection time (see "Timestamps" below)
header.frame_id  -- not consumed programmatically; put something meaningful for your own debugging
                     (e.g. "g1_heading_frame")
point.x          -- ball position, forward axis, meters
point.y          -- ball position, left axis, meters
point.z          -- ball position, up axis, meters. Publish what you actually measure -- do NOT
                     force a constant. See "When this actually matters" below: z naturally comes
                     out ≈ ball radius (e.g. ~0.11) for a resting ball, but a real, moving,
                     above-ground z is expected and IN-distribution once the ball has been struck.
```

**If you don't see the ball, don't publish.** Do not publish a repeated/frozen last-known position,
and do not publish a guess. The whole staleness mechanism downstream (bridge → Redis →
`BallPoseRedisCtrl`, `stale_after_s=0.5` by default) exists specifically so "detector lost track"
correctly reads as **no detection** (the policy gets zeros, its trained fallback) rather than acting
on stale data. Silence is the correct signal for "I don't currently see the ball."

### When this actually matters — task-mode gating

Verified directly against `robojudo/policy/unified_loco_kick_policy.py:385-419`: `kick_ball_pos_b`
is only ever **consumed** while the robot's `task_mode` is `"kick"`. During locomotion (walking up
to the ball, operator-driven via joystick/keyboard velocity commands — this policy does not
autonomously navigate toward the ball at all), the term is unconditionally zeroed in the
observation regardless of what you publish. It becomes live the instant a kick is triggered, and
stays live **every tick, continuously, for the entire kick motion** — not captured once at the
trigger moment. That includes the period *after* foot-ball contact, while the ball is airborne or
rolling away: training's own simulated ball is a real rigid body that gets physically struck and
keeps moving while `task_mode` is still `"kick"` (it's only reset once the mode flips back to
locomotion — see `BallRespawner` in `run_pipeline_prepared.py`). A changing, above-ground `z`
during that window is expected, not a fault.

Practical implication: detector accuracy matters most starting the instant a kick is triggered and
continuously through the strike — that's the only window the policy is actually reading. Getting
`z` (or anything else) slightly wrong during the walk-up approach has no effect on this checkpoint.

### Example

Ball detected 2.0 m directly in front of the robot, 0.3 m to the robot's **right**, resting on the
ground (radius ≈ 0.11 m). Since **right is −y** (y = left):

```python
msg = PointStamped()
msg.header.stamp = node.get_clock().now().to_msg()
msg.header.frame_id = "g1_heading_frame"
msg.point.x = 2.0
msg.point.y = -0.3
msg.point.z = 0.11
ball_pub.publish(msg)
```

## `/kick_aim` — optional, checkpoint-dependent

Only needed if the currently-loaded ONNX checkpoint was trained with `SkillConfig.kick_aim_enabled
= True`. **Check before wiring this up** — see [Checking kick_aim_enabled](#-checking-whether-your-checkpoint-needs-kick_aim)
below. If not needed, skip this topic entirely; `robojudo` defaults `kick_target_pos_b` to `[0, 0]`.

| | |
|---|---|
| **Type** | `geometry_msgs/Vector3Stamped` |
| **Rate** | Low — this is a **command**, not a continuous measurement (see below). A few Hz, or even just once per kick attempt, is enough. |
| **QoS** | Same as `/ball_pose` |
| **Frame** | N/A — this is a bounded scalar bearing offset, not a position |

Fields used (`vector.z` unused):

```
vector.x  -- kick_aim_theta_deg / kick_aim_theta_ref_deg   (normalized, see example)
vector.y  -- always 0.0
```

`kick_aim_theta_ref_deg = 45.0` and `kick_aim_theta_max_deg = 15.0` are project-wide constants (as
of the 2026-08-24 checkpoints) — don't change these unless you know the checkpoint's own
`MultiSkillConfig`/`BallConfig` used different values. `kick_aim_theta_deg` itself is a **bounded
bearing offset** from the skill's own calibrated nominal aim, in **[-15°, +15°]** — it is held
**constant for the whole kick attempt**, not updated tick-to-tick like `/ball_pose`. It comes from
your task/operator layer (e.g. "aim 5° left of center"), not from continuous perception.

### Example

Aim 5° left of the skill's nominal bearing:

```python
theta_deg = 5.0
theta_ref_deg = 45.0
msg = Vector3Stamped()
msg.header.stamp = node.get_clock().now().to_msg()
msg.vector.x = theta_deg / theta_ref_deg   # == 0.1111
msg.vector.y = 0.0
aim_pub.publish(msg)
```

## Timestamps

Set `header.stamp` to the actual detection time (`node.get_clock().now().to_msg()` at the moment
you have a fresh reading — not construction time of an old message, not a fixed value). The bridge
uses this to compute staleness end-to-end; an all-zero/unset stamp is tolerated (falls back to the
bridge's own arrival time) but real detection time is what makes the staleness check honest.

## Coordinate frame — details

The "robot heading frame" is **not** the full 3D body frame — it's a **yaw-only** projection: take
the robot's current heading (which way its torso is pointing, ignoring roll/pitch), and express the
ball's position relative to that, with a level ground-parallel x/y and world-up z. Concretely (this
is the exact transform `robojudo` itself uses, from `robojudo/utils/util_func.py`):

1. Get the robot's current world-frame orientation quaternion.
2. Extract only the yaw component (`heading_quat` — a pure rotation about world z).
3. `ball_pos_heading_frame = inverse(heading_quat) applied to (ball_pos_world - robot_pos_world)`.

If your detector is a camera/lidar rigidly mounted on the robot, you already have the ball's
position relative to the sensor for free — you additionally need the robot's current base
orientation (from the robot's own state estimator / IMU) to project into the yaw-only heading
frame rather than the sensor's own tilted frame. If your sensor mount is roughly level with the
ground when the robot is upright, this may already be close to correct without extra
transformation — but verify, don't assume, especially once the robot is walking (torso pitch/roll
during gait will tilt a naively-used sensor frame away from the yaw-only heading frame the policy
expects).

---

# 🎯 Checking whether your checkpoint needs `kick_aim`

From your workstation (not the robot), with `conda activate robojudo`:

```bash
python -c "
import onnxruntime as ort, json
s = ort.InferenceSession('<path-to-your-.onnx>', providers=['CPUExecutionProvider'])
cfg = json.loads(s.get_modelmeta().custom_metadata_map['experiment_config'])
print('kick_aim_enabled:', cfg['simulator']['config']['scene']['ball']['kick_aim_enabled'])
print('kick_aim_theta_ref_deg:', cfg['simulator']['config']['scene']['ball']['kick_aim_theta_ref_deg'])
print('kick_aim_theta_max_deg:', cfg['simulator']['config']['scene']['ball']['kick_aim_theta_max_deg'])
"
```

If `kick_aim_enabled: True`, wire up `/kick_aim`. If `False` (or the checkpoint predates this
field), skip it.

---

# 🛠️ Step-by-Step: Robot-Side Setup

All of this runs **on the robot's onboard compute, under its native ROS2 Foxy** — not inside
`conda activate robojudo`, which is a completely separate Python (3.12) on a different distro
(Humble). Don't mix the two up when opening terminals.

## 1. Confirm ROS2 Foxy is live

```bash
source /opt/ros/foxy/setup.bash   # or however your robot's ROS2 install is normally sourced
ros2 --help                        # should print usage, not "command not found"
python3 -c "import rclpy; print('rclpy OK')"
```

If your robot sources ROS2 differently (custom install path, a wrapper script, etc.), use whatever
your existing onboard perception/driver stack already uses — the bridge just needs to run under
that same environment.

## 2. Install the bridge's one dependency

The bridge is deliberately minimal — no `robojudo` import, no torch, nothing that risks a repeat
of the C++ ABI issues we hit trying to reconstruct Foxy elsewhere. Just:

```bash
pip install redis
python3 -c "import redis; print('redis client OK')"
```

## 3. Get the bridge script onto the robot

```bash
# from wherever RoboJuDo lives on the robot's filesystem (git clone or already checked out)
git pull   # picks up scripts/foxy_ros2_ball_bridge.py
```

If the robot's onboard compute doesn't have the full `RoboJuDo` checkout, you only need this one
self-contained file — copy it over directly (`scp scripts/foxy_ros2_ball_bridge.py <robot>:...`).

## 4. Decide where Redis runs

`robojudo`'s default (`redis_host="localhost"`) assumes Redis is reachable at `localhost` **from
wherever `run_pipeline_prepared.py` runs**. Two cases:

- **Detector and `robojudo` on the same onboard computer:** run Redis there, everything defaults to
  `localhost`, nothing else to configure.
- **Detector runs on a different machine** (e.g. a separate perception computer, or `robojudo` run
  from your workstation over Ethernet per `unitree_setup.md`'s "Deploy from Your Computer" option):
  Redis needs to be reachable across that link. By default `redis-server` only binds to loopback —
  you'll need to either run Redis on the `robojudo` host and point the bridge's `--redis-host` at
  it, or reconfigure Redis to bind to a real network interface (`redis-server --bind
  <robojudo-host-ip>`) and point `run_pipeline_prepared.py`'s `--redis-host` at that instead. Prefer
  the first (Redis colocated with `robojudo`) — it's one fewer thing to reconfigure.

Install Redis if it isn't already present:

```bash
sudo apt install redis-server   # or: conda install -c conda-forge redis-server
redis-cli ping                   # should print PONG
```

## 5. Run the bridge

```bash
source /opt/ros/foxy/setup.bash
cd /path/to/RoboJuDo
python3 scripts/foxy_ros2_ball_bridge.py
# add --redis-host <ip> if Redis isn't on this same machine (see step 4)
```

Expected log output:
```
[INFO] Connected to redis at localhost:6379/0
[INFO] Subscribed to '/ball_pose' (PointStamped) and '/kick_aim' (Vector3Stamped) -- relaying to redis key 'ball_pose_g1' ...
```

It will sit idle (no further logs) until your detector actually starts publishing — that's normal.

## 6. Build and run your detector node

Minimal template — replace `detect_ball()` with your real camera/lidar logic. Save as e.g.
`my_ball_detector.py` and run it the same way as the bridge (native Foxy environment):

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PointStamped


def detect_ball():
    """Replace with real detection. Return (x, y, z) in the robot heading frame (meters), or
    None if the ball isn't currently visible/tracked -- returning None means "don't publish",
    which is the correct behavior, not an error to work around."""
    raise NotImplementedError


class MyBallDetector(Node):
    def __init__(self):
        super().__init__('my_ball_detector')
        qos = QoSProfile(depth=5, history=HistoryPolicy.KEEP_LAST,
                          reliability=ReliabilityPolicy.BEST_EFFORT,
                          durability=DurabilityPolicy.VOLATILE)
        self.pub = self.create_publisher(PointStamped, '/ball_pose', qos)
        self.create_timer(1.0 / 30.0, self.tick)  # ~30 Hz

    def tick(self):
        result = detect_ball()
        if result is None:
            return  # no detection this tick -- publish nothing, do not fabricate a value
        x, y, z = result
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'g1_heading_frame'
        msg.point.x, msg.point.y, msg.point.z = float(x), float(y), float(z)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = MyBallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## 7. Verify, layer by layer

Check each hop independently before trusting the whole chain — this makes it obvious which layer
is broken if something isn't working, rather than guessing across the whole pipeline at once.

**a. Your detector, in isolation** (bridge doesn't need to be running yet):
```bash
ros2 topic hz /ball_pose      # should show ~30 Hz once your detector sees the ball
ros2 topic echo /ball_pose    # inspect actual values -- sanity-check the sign/magnitude
                               # against where the ball actually is relative to the robot
```

**b. The bridge is relaying correctly:**
```bash
redis-cli get ball_pose_g1
# expect: {"kick_ball_pos_b": [x, y, z], "kick_target_pos_b": [ax, ay], "t": <unix time>}
```

**c. `robojudo` reads it correctly** (from the `robojudo` host, `conda activate robojudo`):
```bash
python -c "
from robojudo.controller.ball_pose_redis_ctrl import BallPoseRedisCtrl
from robojudo.controller.ctrl_cfgs import BallPoseRedisCtrlCfg
ctrl = BallPoseRedisCtrl(cfg_ctrl=BallPoseRedisCtrlCfg(), env=None)
print(ctrl.get_data())   # valid=True, kick_ball_pos_b matching what you saw in step (a)
"
```

**d. Full run:**
```bash
python scripts/run_pipeline_prepared.py -c g1_unified_loco_kick --live-ball --ball-source redis
```

---

# 🩺 Troubleshooting

- **`redis-cli get ball_pose_g1` returns nothing / `(nil)`** — the bridge isn't running, or your
  detector isn't publishing yet, or Redis network/host mismatch (see step 4). Check layer (a) and
  (b) above in order.
- **`valid: False` from `BallPoseRedisCtrl.get_data()` despite Redis having a value** — the `t`
  field is older than `stale_after_s` (0.5s default). Usually means your detector's publish rate is
  too low, has stalled, or the bridge lost its subscription. Check `ros2 topic hz /ball_pose`.
- **Robot walks toward the wrong spot** — almost always a frame/sign bug: either the y-axis sign is
  flipped (remember: **y = left**, so an object to the robot's right is **negative** y), or the
  heading-frame yaw projection wasn't applied and you're publishing in the raw sensor frame instead
  (see [Coordinate frame](#-coordinate-frame-details)). Verify with `ros2 topic echo /ball_pose`
  while manually placing the ball at a known position relative to the robot.
- **`kick_target_pos_b` always `[0, 0]` even though `/kick_aim` is being published** — confirm the
  bridge actually has an `/kick_aim` subscription connected: `ros2 topic info /kick_aim
  --verbose` should show both a publisher and a subscriber. A QoS mismatch (rare, since the bridge
  subscribes at the most permissive setting) or a topic name typo are the usual causes.
- **Detector import/build issues under Foxy** — keep the detector (and the bridge) as dependency-
  light as possible. Foxy is EOL and its ecosystem is increasingly hard to extend without hitting
  the same C++ ABI fragility that made a reconstructed Foxy unusable for testing (see
  [Architecture](#-architecture--why-it-has-this-shape) above) — avoid pulling in heavy new
  compiled dependencies under Foxy if you can do the heavy lifting (model inference, etc.) in a
  separate process and hand off only the final detection.

---

# 🔭 Future: skipping the bridge

If Humble↔Foxy DDS interoperability is ever confirmed (this needs testing directly against the
robot's real Foxy install — a reconstructed Foxy in an isolated sandbox was not usable for this,
see [Architecture](#-architecture--why-it-has-this-shape) above), your detector could publish directly
to `robojudo`'s own `BallPoseRos2Ctrl` (`--ball-source ros2` on `run_pipeline_prepared.py`) with
**zero changes** — it uses the exact same `/ball_pose` + `/kick_aim` topic/type/QoS contract this
doc describes. The bridge would become optional, not a rewrite.
