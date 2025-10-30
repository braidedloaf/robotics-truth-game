#!/usr/bin/env python3
"""
Launcher for the blocks + buckets + MediaPipe claw demo.

Usage examples:
  python start.py
  python start.py --cam 1
  python start.py --width 1280 --height 720 --no-mirror
  python start.py --max-speed 2200 --grab-force 300000
"""

import sys
import argparse

# Dependency preflight with clear errors
def _require(pkgs):
    missing = []
    for name in pkgs:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install with:\n  pip install -r requirements.txt")
        sys.exit(1)

_require(["pygame", "pymunk", "cv2", "mediapipe", "numpy"])

# Import the main app module (must be in the same directory)
try:
    import block_sim as app
except Exception as e:
    print("Error: could not import 'blocks_sim.py'. Make sure it is in the same directory.")
    print("Python exception:", repr(e))
    sys.exit(1)

def parse_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--cam", type=int, default=0, help="Camera index for OpenCV (default 0).")
    p.add_argument("--width", type=int, default=None, help="Window width in pixels.")
    p.add_argument("--height", type=int, default=None, help="Window height in pixels.")
    p.add_argument("--no-mirror", action="store_true", help="Disable camera mirroring.")
    p.add_argument("--fps", type=int, default=None, help="Display FPS cap.")
    p.add_argument("--dt", type=float, default=None, help="Fixed physics dt (e.g., 0.0033 for ~300 Hz).")
    p.add_argument("--max-speed", type=float, default=None, help="Max claw speed (px/s).")
    p.add_argument("--max-acc", type=float, default=None, help="Max claw acceleration (px/s^2).")
    p.add_argument("--grab-force", type=float, default=None, help="Pivot joint max force.")
    p.add_argument("--spring-k", type=float, default=None, help="Translational spring stiffness.")
    p.add_argument("--spring-c", type=float, default=None, help="Translational spring damping.")
    return p.parse_args()

def apply_overrides(args):
    # Resolution
    if args.width is not None:
        app.SCREEN_W = int(args.width)
    if args.height is not None:
        app.SCREEN_H = int(args.height)

    # Derived geometry that depends on width/height
    if hasattr(app, "FLOOR_Y"):
        app.FLOOR_Y = app.SCREEN_H - 60
    if hasattr(app, "BUCKET_W"):
        app.BUCKET_W = app.SCREEN_W // 6
    if hasattr(app, "BUCKET_H"):
        app.BUCKET_H = app.SCREEN_H // 2

    # Camera mirroring
    if args.no_mirror:
        if hasattr(app, "CAM_MIRROR"):
            app.CAM_MIRROR = False

    # Timing and solver knobs
    if args.fps is not None and hasattr(app, "FPS_CAP"):
        app.FPS_CAP = int(args.fps)
    if args.dt is not None and hasattr(app, "DT_FIXED"):
        app.DT_FIXED = float(args.dt)

    # Claw dynamics
    if args.max_speed is not None and hasattr(app, "MAX_CLAW_SPEED"):
        app.MAX_CLAW_SPEED = float(args.max_speed)
    if args.max_acc is not None and hasattr(app, "MAX_CLAW_ACC"):
        app.MAX_CLAW_ACC = float(args.max_acc)

    # Gripper strength
    if args.grab_force is not None and hasattr(app, "GRAB_FORCE"):
        app.GRAB_FORCE = float(args.grab_force)
    if args.spring_k is not None and hasattr(app, "SPRING_K"):
        app.SPRING_K = float(args.spring_k)
    if args.spring_c is not None and hasattr(app, "SPRING_C"):
        app.SPRING_C = float(args.spring_c)

    # Pass camera index into the tracker by patching its constructor default if present
    # Fallback: the app’s HandTracker reads default cam_index=0; we wrap it if needed.
    if getattr(app.HandTracker.__init__, "__code__", None) and "cam_index" in app.HandTracker.__init__.__code__.co_varnames:
        # Monkey-patch a factory so app.main() uses the right index if it constructs internally.
        original_cls = app.HandTracker
        cam_idx = int(args.cam)

        class _HT(original_cls):  # type: ignore
            def __init__(self, cam_index: int = cam_idx):
                super().__init__(cam_index=cam_index)

        app.HandTracker = _HT  # replace in module namespace

def main():
    args = parse_args()
    apply_overrides(args)
    # Run app
    try:
        app.main()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
