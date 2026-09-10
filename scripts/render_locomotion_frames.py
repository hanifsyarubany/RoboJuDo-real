#!/usr/bin/env python3
"""Render the deployed locomotion policy to transparent-background PNG frames.

Rolls out the real sim2sim stack -- same MuJoCo env, same ONNX checkpoint, same
command rate-limiter as scripts/run_pipeline_prepared.py -- but drives the
velocity command programmatically instead of from a keyboard/joystick, and
writes one RGBA PNG per output frame with the robot cut out of its background.

Usage:
    # all four default clips (stand / walk / strafe / rotate)
    python scripts/render_locomotion_frames.py

    # a subset, longer, at 25 fps
    python scripts/render_locomotion_frames.py --clips walk rotate --seconds 6 --fps 25

    # a custom command: name:vx,vy,wz  (m/s, m/s, rad/s; +x fwd, +y left, +z CCW)
    python scripts/render_locomotion_frames.py --clip dash:0.8,0,0 --clip back:-0.4,0,0

    # flat background instead of transparency, plus an mp4 per clip
    python scripts/render_locomotion_frames.py --bg-color white --video

Runs headless: the env's interactive viewer is closed immediately after
construction (it is also what makes stepping fast) and frames come from a
separate offscreen mujoco.Renderer.

No Redis and no ball process are involved -- this is the locomotion path only,
so the policy stays in task_mode=locomotion for the whole rollout.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np

# import robojudo.pipeline (and therefore torch) BEFORE mujoco -- see the same note in
# scripts/run_pipeline_prepared.py about glibc static TLS exhaustion on the onboard computer.
import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager

import mujoco  # noqa: E402 -- see note above

logger = logging.getLogger("robojudo")

# name -> (vx, vy, wz) in the policy's command units: m/s forward, m/s left, rad/s CCW.
DEFAULT_CLIPS = {
    "stand": (0.0, 0.0, 0.0),
    "walk": (0.6, 0.0, 0.0),
    "strafe": (0.0, 0.4, 0.0),
    "rotate": (0.0, 0.0, 0.6),
}

NAMED_COLORS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "grey": (128, 128, 128),
    "gray": (128, 128, 128),
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
}

FALL_BASE_Z = 0.4  # below this the robot is down, not walking -- stop the clip


# --------------------------------------------------------------------------
# command injection
# --------------------------------------------------------------------------
def axis_for(velocity: float, row) -> float:
    """Invert command_remap: the joystick axis value that yields `velocity`.

    commands_map rows are [min, mid, max] and are NOT all increasing (the
    lateral and yaw rows are inverted), so pick the branch by sign rather than
    assuming max > min.
    """
    lo, mid, hi = (float(v) for v in row)
    scale_pos, scale_neg = hi - mid, mid - lo
    axis = (velocity - mid) / scale_pos if scale_pos else 0.0
    if axis <= 0:
        axis = (velocity - mid) / scale_neg if scale_neg else 0.0
    return float(np.clip(axis, -1.0, 1.0))


def install_command_source(pipeline) -> dict:
    """Feed the policy a synthetic joystick reading. Returns the mutable axes dict.

    Injected as the FIRST key and with the real human controllers dropped:
    _update_velocity_command breaks on whichever of JoystickCtrl / UnitreeCtrl /
    KeyboardCtrl it meets first, so a real KeyboardCtrl left ahead of ours would
    win and report a permanent zero command. Everything else (ball controllers)
    is passed through untouched.
    """
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


def set_command(axes: dict, policy, vx: float, vy: float, wz: float) -> None:
    axes["LeftY"] = axis_for(vx, policy.commands_map[0])
    axes["LeftX"] = axis_for(vy, policy.commands_map[1])
    axes["RightX"] = axis_for(wz, policy.commands_map[2])


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------
def downsample(img, scale: int):
    """Box-average a supersampled render down, premultiplying alpha first.

    Without premultiplication the background colour bleeds into edge pixels and
    leaves a halo once alpha is applied.
    """
    if scale <= 1:
        return img

    arr = img.astype(np.float32)
    has_alpha = arr.shape[2] == 4
    if has_alpha:
        a = arr[..., 3:4] / 255.0
        arr = np.dstack([arr[..., :3] * a, arr[..., 3:4]])

    h, w, c = arr.shape
    small = arr.reshape(h // scale, scale, w // scale, scale, c).mean(axis=(1, 3))

    if has_alpha:
        a = small[..., 3:4] / 255.0
        rgb = np.divide(small[..., :3], a, out=np.zeros_like(small[..., :3]), where=a > 1e-6)
        small = np.dstack([rgb, small[..., 3:4]])

    return np.clip(small, 0, 255).astype(np.uint8)


def parse_color(spec: str):
    if spec in NAMED_COLORS:
        return NAMED_COLORS[spec]
    try:
        r, g, b = (int(v) for v in spec.split(","))
    except ValueError:
        raise SystemExit(
            f"[Error] Bad colour '{spec}'. Use R,G,B or one of: {', '.join(NAMED_COLORS)}"
        ) from None
    return r, g, b


def composite(img, color):
    """Flatten an RGBA cutout onto a solid colour. RGB passes through."""
    if img.shape[2] == 3:
        return img
    rgb = parse_color(color) if isinstance(color, str) else color
    a = img[..., 3:4].astype(np.float32) / 255.0
    flat = img[..., :3].astype(np.float32) * a + np.float32(rgb) * (1.0 - a)
    return np.clip(flat, 0, 255).astype(np.uint8)


def draw_label(img, text: str):
    from PIL import Image, ImageDraw

    pil = Image.fromarray(img)
    rgba = pil.mode == "RGBA"
    box = (0, 0, 0) + ((255,) if rgba else ())
    ink = (255, 255, 255) + ((255,) if rgba else ())
    draw = ImageDraw.Draw(pil)
    draw.rectangle([0, 0, 8 + 6 * len(text), 18], fill=box)
    draw.text((4, 4), text, fill=ink)
    return np.asarray(pil)


def make_contact_sheet(frames, out_path: Path, cols: int, max_tiles: int) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    picks = frames
    if len(frames) > max_tiles:
        step = np.linspace(0, len(frames) - 1, max_tiles).round().astype(int)
        picks = [frames[i] for i in step]

    tiles = []
    for path in picks:
        tile = Image.open(path)
        if tile.mode == "RGBA":  # a cutout would otherwise tile as black
            tile = Image.alpha_composite(Image.new("RGBA", tile.size, (255,) * 4), tile)
        tile = tile.convert("RGB")
        tile.thumbnail((480, 480))
        label = path.stem.replace("frame_", "f")
        draw = ImageDraw.Draw(tile)
        draw.rectangle([0, 0, 8 + 6 * len(label), 16], fill=(40, 40, 40))
        draw.text((4, 3), label, fill=(255, 255, 255))
        tiles.append(np.asarray(tile))

    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * th, cols * tw, 3), 255, dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th : r * th + tile.shape[0], c * tw : c * tw + tile.shape[1]] = tile

    imageio.imwrite(out_path, sheet)
    print(f"[Render] contact sheet ({len(tiles)} tiles) -> {out_path}")


# --------------------------------------------------------------------------
def render_clip(pipeline, axes, renderer, cam, name, command, out_dir: Path, args) -> list[Path]:
    """Reset, settle at zero command, then hold `command` while writing frames."""
    import imageio.v2 as imageio

    env = pipeline.env
    policy = pipeline.policy.policy

    # start every clip from the same grounded stance, exactly as run_pipeline_prepared.py does
    mujoco.mj_resetDataKeyframe(env.model, env.data, 0)
    mujoco.mj_forward(env.model, env.data)
    env.update()
    pipeline.reset()
    policy.reset()

    set_command(axes, policy, 0.0, 0.0, 0.0)
    for _ in range(int(round(args.settle / pipeline.dt))):
        pipeline.step()

    set_command(axes, policy, *command)
    stride = max(1, int(round((1.0 / pipeline.dt) / args.fps)))
    n_ticks = int(round(args.seconds / pipeline.dt))
    robot_geoms = np.flatnonzero(env.model.geom_bodyid != 0)
    base_id = env.model.body(args.base_body).id if args.base_body else None

    # Roll the whole clip out first, then render from the recorded states. Rendering as we step
    # would work for a tracking camera, but a fixed camera has to be framed from the trajectory
    # it does not yet know -- and stepping is cheap next to rendering anyway.
    poses: list[np.ndarray] = []
    fell_at = None
    for tick in range(n_ticks):
        pipeline.step()
        if env.data.qpos[2] < FALL_BASE_Z:
            fell_at = tick * pipeline.dt
            break
        if tick % stride == 0:
            poses.append(env.data.qpos.copy())

    if not args.follow and poses:
        # frame the whole traversal instead of tracking the pelvis, so that walking and
        # strafing read as movement rather than as a gait cycle in place
        base_xyz = np.array([q[:3] for q in poses])
        cam.lookat[:] = base_xyz.mean(axis=0)
        span = np.ptp(base_xyz, axis=0)
        cam.distance = args.distance + 0.75 * float(np.linalg.norm(span[:2]))

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for qpos in poses:
        env.data.qpos[:] = qpos
        mujoco.mj_forward(env.model, env.data)

        if args.follow:
            cam.lookat[:] = env.data.xpos[base_id] if base_id is not None else env.data.qpos[:3]
        renderer.update_scene(env.data, camera=cam)
        img = renderer.render()

        if args.cutout:
            renderer.enable_segmentation_rendering()
            renderer.update_scene(env.data, camera=cam)
            seg = renderer.render()
            renderer.disable_segmentation_rendering()
            alpha = np.isin(seg[..., 0], robot_geoms).astype(np.uint8) * 255
            img = downsample(np.dstack([img, alpha]), args.supersample)
            if args.bg_color is not None:
                img = composite(img, args.bg_color)
        else:
            img = downsample(img, args.supersample)

        if args.label:
            img = draw_label(img, f"{name}  t={len(written) / args.fps:.2f}s  "
                                  f"cmd=({command[0]:+.1f},{command[1]:+.1f},{command[2]:+.1f})")

        path = out_dir / f"frame_{len(written):05d}.png"
        imageio.imwrite(path, img)
        written.append(path)

    cam.distance = args.distance  # --no-follow widens it per clip; do not leak into the next one

    travel = float(np.linalg.norm(poses[-1][:2] - poses[0][:2])) if len(poses) > 1 else 0.0
    applied = (policy.lin_vel_command[0], policy.lin_vel_command[1], policy.ang_vel_command)
    print(
        f"[Render] {name}: {len(written)} frames @ {args.fps} fps -> {out_dir}\n"
        f"         commanded ({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}) "
        f"| applied ({applied[0]:+.2f}, {applied[1]:+.2f}, {applied[2]:+.2f}) "
        f"| travelled {travel:.2f} m | final base_z {env.data.qpos[2]:.3f} | task_mode {policy.task_mode}"
    )
    if fell_at is not None:
        print(f"[Warn]   {name}: base dropped below {FALL_BASE_Z} m at t={fell_at:.2f}s -- clip cut short")

    if args.video and written:
        video_path = out_dir / "_motion.mp4"
        try:
            with imageio.get_writer(video_path, fps=args.fps) as writer:
                for path in written:  # mp4 carries no alpha, so a cutout has to be flattened
                    writer.append_data(composite(imageio.imread(path), args.bg_color or "white"))
            print(f"[Render] video -> {video_path}")
        except ValueError as exc:
            # the PNGs are the deliverable -- never lose a finished rollout to a missing encoder
            video_path.unlink(missing_ok=True)
            print(f"[Warn]   {name}: mp4 skipped ({exc.__class__.__name__}: no ffmpeg backend in this "
                  f"env). Install it with: pip install imageio-ffmpeg")

    return written


def parse_clip_specs(args) -> dict:
    clips = {}
    for name in args.clips:
        if name not in DEFAULT_CLIPS:
            raise SystemExit(
                f"[Error] Unknown clip '{name}'. Known: {', '.join(DEFAULT_CLIPS)} "
                f"(or define one with --clip name:vx,vy,wz)"
            )
        clips[name] = DEFAULT_CLIPS[name]

    for spec in args.clip:
        try:
            name, values = spec.split(":", 1)
            vx, vy, wz = (float(v) for v in values.split(","))
        except ValueError:
            raise SystemExit(f"[Error] Bad --clip '{spec}'. Expected name:vx,vy,wz") from None
        clips[name] = (vx, vy, wz)
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the locomotion policy to transparent PNG frames (headless)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-c", "--config", default="g1_unified_loco_kick", help="RoboJuDo config name")
    parser.add_argument("--onnx-path", default=None, help="Override the checkpoint in the policy config")
    parser.add_argument("--clips", nargs="*", default=list(DEFAULT_CLIPS),
                        help=f"Built-in clips to render (default: all of {', '.join(DEFAULT_CLIPS)})")
    parser.add_argument("--clip", action="append", default=[], metavar="NAME:VX,VY,WZ",
                        help="Extra clip with an explicit command (m/s, m/s, rad/s; +x fwd, +y left, +z CCW)")

    parser.add_argument("--seconds", type=float, default=4.0, help="Recorded length per clip (default: 4)")
    parser.add_argument("--settle", type=float, default=1.0,
                        help="Seconds held at zero command before recording (default: 1)")
    parser.add_argument("--fps", type=float, default=25.0, help="Output frame rate (default: 25; control loop is 50 Hz)")
    parser.add_argument("--out", default="logs/locomotion_frames", help="Output directory")

    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-10.0)
    parser.add_argument("--distance", type=float, default=2.5)
    parser.add_argument("--base-body", default="pelvis", help="Body the camera tracks ('' = base qpos)")
    parser.add_argument("--no-follow", dest="follow", action="store_false",
                        help="Fix the camera and frame the whole traversal, so translation is visible")

    parser.add_argument("--no-cutout", dest="cutout", action="store_false",
                        help="Keep the floor and sky instead of cutting the robot out")
    parser.add_argument("--bg-color", default=None,
                        help="Composite the cutout onto a flat colour instead of transparency")
    parser.add_argument("--supersample", type=int, default=2, help="Render NxN larger and average down (default: 2)")

    parser.add_argument("--label", action="store_true", help="Burn clip name / time / command into each PNG")
    parser.add_argument("--contact-sheet", action="store_true", help="Also write one overview grid per clip")
    parser.add_argument("--sheet-cols", type=int, default=6)
    parser.add_argument("--sheet-tiles", type=int, default=24)
    parser.add_argument("--video", action="store_true", help="Also write an mp4 per clip")

    args = parser.parse_args()
    if args.bg_color is not None:
        parse_color(args.bg_color)  # fail now, not mid-rollout
    if args.supersample < 1:
        parser.error("--supersample must be >= 1")
    clips = parse_clip_specs(args)
    if not clips:
        parser.error("no clips selected")

    os.environ.setdefault("MUJOCO_GL", "egl")

    cfg = ConfigManager(config_name=args.config).get_cfg()
    if args.onnx_path:
        cfg.policy.onnx_path = args.onnx_path
    if not cfg.env.is_sim:
        raise SystemExit(f"[Error] Config '{args.config}' is not a sim config -- refusing to drive real hardware.")

    pipeline = getattr(robojudo.pipeline, cfg.pipeline_type)(cfg=cfg)
    env = pipeline.env
    # the interactive viewer needs a display and dominates the step cost; frames come from our
    # own offscreen renderer instead (env.step() skips rendering once it is not alive)
    env.viewer.close()

    if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_KEY, "default_stand") < 0:
        raise SystemExit("[Error] sim env has no 'default_stand' keyframe -- see scene_g1_29dof.xml")

    scale = args.supersample
    env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, args.width * scale)
    env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, args.height * scale)
    renderer = mujoco.Renderer(env.model, height=args.height * scale, width=args.width * scale)

    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance

    axes = install_command_source(pipeline)
    out_root = Path(args.out)
    print(f"[Setup] checkpoint: {cfg.policy.onnx_path}")
    print(f"[Setup] {1 / pipeline.dt:.0f} Hz control, {args.fps:g} fps out, "
          f"{args.width}x{args.height}, {'RGBA cutout' if args.cutout and args.bg_color is None else 'flat'}")

    for name, command in clips.items():
        render_clip(pipeline, axes, renderer, cam, name, command, out_root / name, args)
        if args.contact_sheet:
            frames = sorted((out_root / name).glob("frame_*.png"))
            if frames:
                make_contact_sheet(frames, out_root / name / "_contact_sheet.png",
                                   args.sheet_cols, args.sheet_tiles)

    renderer.close()
    print(f"[Done] {len(clips)} clip(s) in {out_root}")


if __name__ == "__main__":
    main()
