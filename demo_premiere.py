#!/usr/bin/env python3
"""Cinematic UR10 motion-sim demo — records a premiere-style MP4.

Smooth joint trajectories + fixed scene camera (not wrist POV). Does not
train or collect VLA data — visual demo only.

Usage
-----
    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u demo_premiere.py [--headless]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--out", default="docs/premiere_motion_demo.mp4")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": args.width,
        "height": args.height,
    }
)

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualSphere
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.universal_robots.ur10 import UR10 as _UR10
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path


class UR10(_UR10):
    def post_reset(self) -> None:
        super(_UR10, self).post_reset()
        self._end_effector.post_reset()
        if self._gripper is not None:
            self._gripper.post_reset()


physics_dt = 1.0 / 120.0
render_dt = 1.0 / args.fps
steps_per_frame = max(1, int(round(render_dt / physics_dt)))
n_frames = int(args.seconds * args.fps)

world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
assets = get_assets_root_path()

define_prim("/World/Ground", "Xform").GetReferences().AddReference(
    assets + "/Isaac/Environments/Grid/default_environment.usd"
)

robot = world.scene.add(
    UR10(
        prim_path="/World/UR10",
        name="ur10",
        position=np.array([0.0, 0.0, 0.0]),
        attach_gripper=False,
    )
)

# Accent target — bright cyan so it reads on camera
target = world.scene.add(
    VisualSphere(
        prim_path="/World/Target",
        name="target",
        position=np.array([0.55, 0.25, 0.45]),
        radius=0.035,
        color=np.array([0.1, 0.85, 1.0]),
    )
)

# Premiere-style 3/4 camera (looks at workspace center)
cam = Camera(
    prim_path="/World/CineCam",
    resolution=(args.width, args.height),
    position=np.array([1.55, -1.35, 1.05]),
    orientation=euler_angles_to_quat(np.array([55.0, 0.0, 50.0]), degrees=True),
)

world.reset()
cam.initialize()

# Warm up render / annotators
for _ in range(12):
    world.step(render=True)

# Home + show poses (rad) — smooth cosine blend between them
q_home = np.array([0.0, -1.0, 1.2, -1.4, -1.57, 0.0], dtype=np.float64)
q_reach = np.array([0.55, -0.85, 1.05, -1.55, -1.57, 0.35], dtype=np.float64)
q_sweep = np.array([-0.45, -0.7, 0.95, -1.35, -1.2, -0.4], dtype=np.float64)

# Segment timeline as fractions of total frames
keyframes = [
    (0.00, q_home),
    (0.22, q_reach),
    (0.48, q_sweep),
    (0.72, q_reach),
    (1.00, q_home),
]


def sample_q(frac: float) -> np.ndarray:
    frac = float(np.clip(frac, 0.0, 1.0))
    for i in range(len(keyframes) - 1):
        t0, q0 = keyframes[i]
        t1, q1 = keyframes[i + 1]
        if frac <= t1 or i == len(keyframes) - 2:
            u = 0.0 if t1 <= t0 else (frac - t0) / (t1 - t0)
            u = 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0.0, 1.0))  # ease in-out
            return (1.0 - u) * q0 + u * q1
    return keyframes[-1][1]


frame_dir = tempfile.mkdtemp(prefix="ur_premiere_")
print(f"Recording {n_frames} frames @ {args.fps}fps → {frame_dir}")

controller = robot.get_articulation_controller()
wrote = 0

for fi in range(n_frames):
    frac = fi / max(1, n_frames - 1)
    q_des = sample_q(frac).astype(np.float32)
    # Gentle wrist flutter so motion never looks frozen
    q_des = q_des.copy()
    q_des[5] += 0.12 * np.sin(2.0 * np.pi * frac * 2.0)

    for _ in range(steps_per_frame):
        controller.apply_action(ArticulationAction(joint_positions=q_des))
        world.step(render=True)

    rgb = cam.get_rgb()
    if rgb is None or rgb.size == 0:
        continue
    # Camera returns float [0,1] or uint8 depending on build — normalize
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]

    path = os.path.join(frame_dir, f"frame_{wrote:05d}.png")
    try:
        from PIL import Image

        Image.fromarray(rgb).save(path)
    except ImportError:
        import imageio

        imageio.imwrite(path, rgb)
    wrote += 1
    if fi % args.fps == 0:
        print(f"  {fi}/{n_frames} frames ({100 * frac:.0f}%)")

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
cmd = [
    "ffmpeg",
    "-y",
    "-framerate",
    str(args.fps),
    "-i",
    os.path.join(frame_dir, "frame_%05d.png"),
    "-c:v",
    "libx264",
    "-pix_fmt",
    "yuv420p",
    "-crf",
    "18",
    "-movflags",
    "+faststart",
    args.out,
]
print("Encoding:", " ".join(cmd))
subprocess.run(cmd, check=True)
shutil.rmtree(frame_dir, ignore_errors=True)

print(f"\nPremiere clip ready: {args.out} ({wrote} frames, {args.seconds:.1f}s)")
world.stop()
simulation_app.close()
