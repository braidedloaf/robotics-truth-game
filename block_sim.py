# --- IMPORTS ---
import math
import os.path
import random
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pygame
import pymunk
import pymunk.pygame_util

import cv2
import mediapipe as mp
import numpy as np

pygame.mixer.pre_init(44100, -16, 2, 256)
pygame.init()

# --- CONFIG ---
SCREEN_W = 1360
SCREEN_H = 768
BG_COLOR = (18, 18, 20)
FLOOR_Y = SCREEN_H - 60

GRAVITY_Y = 1600.0

DENSITY = 0.0008
W_RANGE = (40, 100)
H_RANGE = (40, 100)

FPS_CAP = 30                 # display cap
SOLVER_ITER = 60
DT_FIXED = 1.0 / 300.0
GLOBAL_DAMPING = 0.99
SLEEP_THRESHOLD = 0.3

BUCKET_W = SCREEN_W // 6
BUCKET_H = SCREEN_H // 2
BUCKET_WALL = 8
BUCKET_COLOR = (120, 120, 140, 255)

CLAW_RADIUS = 18
CLAW_COLOR_IDLE = (180, 200, 255)
CLAW_COLOR_GRAB = (255, 200, 60)

# True-Collector mode tuning
COLLECT_MAX_GRABS = 6
COLLECT_RADIUS = 48.0
COLLECT_COOLDOWN_MS = 80

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17)              # palm crossbars
]

# Grabbing
GRAB_MAX_DIST = 28
GRAB_FORCE = 2.0e5
SPRING_K  = 8000.0
SPRING_C  = 1200.0

# Camera panel
CAM_MIRROR = True
CAM_W = 480
CAM_H = 270
CAM_POS = (12, 12)

CAM_IDX = 0

CAM_PAD = 12
FLIP_HYSTERESIS = 24

PAD = 40
CEILING_Y = -200

CAM_LEFT = True

# Input cropping
EDGE_MARGIN_X = 0.15
EDGE_MARGIN_Y = 0.10

# Hand filtering and classification
MAX_CLAW_SPEED = 3000.0
MAX_CLAW_ACC   = 90000.0
PINCH_CLOSE = 0.35
PINCH_OPEN = 0.45
HOLD_MISS_T = 0.08
MEDIAN_WIN = 5
OE_MINCUTOFF = .8
OE_BETA = 0.05
OE_DCUTOFF = 1.0

# --- Difficulties -> initial spawn counts ---
DIFFICULTIES = [
    ("Easy",   8),
    ("Medium", 16),
    ("Hard",   28),
]

# Menu layout
MENU_BTN_W = 360
MENU_BTN_H = 90
MENU_BTN_GAP = 24
MENU_SELECT_COOLDOWN = 0.40

# Toggle layout
TOGGLE_W = 320
TOGGLE_H = 56

SND_CLICK = pygame.mixer.Sound(os.path.join("assets", "menu_click.ogg"))
SND_CORRECT = pygame.mixer.Sound(os.path.join("assets", "correct.wav"))
SND_INCORRECT = pygame.mixer.Sound(os.path.join("assets", "incorrect.wav"))

# --- UTIL: One Euro Filter ---
class OneEuro:
    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x_prev = None
        self._dx_prev = 0.0

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    @staticmethod
    def _exp_smooth(prev, x, a):
        return a * x + (1.0 - a) * prev

    def filter(self, x, dt):
        if self._x_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            return x
        dx = (x - self._x_prev) / max(dt, 1e-6)
        a_d = self._alpha(dt, self.dcutoff)
        dx_hat = self._exp_smooth(self._dx_prev, dx, a_d)
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = self._exp_smooth(self._x_prev, x, a)
        self._x_prev, self._dx_prev = x_hat, dx_hat
        return x_hat


# --- HAND INPUT ---
@dataclass
class ClawInput:
    x: float
    y: float
    closed: bool
    present: bool

def menu_button_rects() -> list[pygame.Rect]:
    total_h = len(DIFFICULTIES) * MENU_BTN_H + (len(DIFFICULTIES) - 1) * MENU_BTN_GAP
    x = (SCREEN_W - MENU_BTN_W) // 2
    y0 = (SCREEN_H - total_h) // 2
    rects = []
    for i in range(len(DIFFICULTIES)):
        rects.append(pygame.Rect(x, y0 + i * (MENU_BTN_H + MENU_BTN_GAP), MENU_BTN_W, MENU_BTN_H))
    return rects

def point_in_rect(px: float, py: float, r: pygame.Rect) -> bool:
    return r.collidepoint(int(px), int(py))


def setup_mass_profile():
    global DENSITY, W_RANGE, H_RANGE

    wmin = random.randint(36, 64)
    wmax = random.randint(96, 140)
    hmin = random.randint(36, 64)
    hmax = random.randint(96, 140)
    W_RANGE = (wmin, wmax)
    H_RANGE = (hmin, hmax)

    # target max mass and implied density
    M_max_run = random.uniform(6.0, 18.0)
    A_max = wmax * hmax
    DENSITY = M_max_run / A_max

    Ew = 0.5 * (wmin + wmax)
    Eh = 0.5 * (hmin + hmax)
    E_area = Ew * Eh
    E_mass = DENSITY * E_area

    return {
        "w_range": W_RANGE, "h_range": H_RANGE,
        "max_mass": M_max_run, "avg_mass": E_mass, "density": DENSITY
    }



def _crop_map(n, lo, hi):
    """
    Map normalized n in [0,1] using a crop [lo, 1-hi] back to [0,1].
    Values outside crop clamp to edges.
    """
    lo = max(0.0, min(0.49, lo))
    hi = max(0.0, min(0.49, hi))
    span = 1.0 - lo - hi
    if span <= 1e-6:
        return 0.5
    n = (n - lo) / span
    return 0.0 if n < 0.0 else 1.0 if n > 1.0 else n

def trigger_color_flash(sim: "PhysicsSim", shape: pymunk.Shape, rgb: tuple[int,int,int], duration_ms: int = 300):
    now = pygame.time.get_ticks()
    if shape not in sim._orig_color:
        sim._orig_color[shape] = shape.color
    shape.color = (*rgb, 255)
    sim._flash_expire[shape] = now + duration_ms

def update_color_flashes(sim: "PhysicsSim"):
    now = pygame.time.get_ticks()
    for shp, t_exp in list(sim._flash_expire.items()):
        if now >= t_exp:
            orig = sim._orig_color.get(shp)
            if orig is not None:
                shp.color = orig
            del sim._flash_expire[shp]

class HandTracker:
    """
    MediaPipe hands -> screen coords + open or closed.
    Jitter reduction: median and One Euro filtering.
    """
    def __init__(self, cam_index: int = 0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not found or cannot be opened.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 270)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.last_rgb = None
        self._median_x = deque(maxlen=MEDIAN_WIN)
        self._median_y = deque(maxlen=MEDIAN_WIN)
        self._euro_x = OneEuro(OE_MINCUTOFF, OE_BETA, OE_DCUTOFF)
        self._euro_y = OneEuro(OE_MINCUTOFF, OE_BETA, OE_DCUTOFF)
        self._closed_state = False
        self._last_seen_t = 0.0

    @staticmethod
    def _median(vals: deque) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    def read(self, dt_frame: float) -> ClawInput:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.last_rgb = None
            self.last_pairs = []
            self._last_seen_t += dt_frame
            present = self._last_seen_t <= HOLD_MISS_T
            x = self._euro_x._x_prev if self._euro_x._x_prev is not None else SCREEN_W / 2
            y = self._euro_y._x_prev if self._euro_y._x_prev is not None else 0.0
            return ClawInput(float(x), float(y), self._closed_state, present)

        if CAM_MIRROR:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = True
        res = self.hands.process(rgb)

        self.last_pairs = []
        h, w, _ = rgb.shape

        if res.multi_hand_landmarks:
            lms = res.multi_hand_landmarks
            hds = getattr(res, "multi_handedness", None)

            for idx, lm in enumerate(lms):
                label = None
                if hds and len(hds) > idx and getattr(hds[idx], "classification", None):
                    label = hds[idx].classification[0].label

                xs = np.fromiter((p.x for p in lm.landmark), dtype=np.float32, count=21)
                ys = np.fromiter((p.y for p in lm.landmark), dtype=np.float32, count=21)

                x_min, x_max = float(xs.min()), float(xs.max())
                y_min, y_max = float(ys.min()), float(ys.max())
                cx_n = 0.5 * (x_min + x_max)
                cy_n = 0.5 * (y_min + y_max)

                for a, b in HAND_CONN:
                    ax, ay = int(xs[a] * w), int(ys[a] * h)
                    bx, by = int(xs[b] * w), int(ys[b] * h)
                    cv2.line(rgb, (ax, ay), (bx, by), (80, 220, 255), 1, cv2.LINE_AA)

                for i in range(21):
                    jx, jy = int(xs[i] * w), int(ys[i] * h)
                    cv2.circle(rgb, (jx, jy), 2, (255, 230, 80), -1, cv2.LINE_AA)

                x0, y0 = int(x_min * w), int(y_min * h)
                x1, y1 = int(x_max * w), int(y_max * h)
                cv2.rectangle(rgb, (x0, y0), (x1, y1), (255, 220, 80), 1, cv2.LINE_AA)
                if label:
                    cv2.putText(rgb, label, (x0, max(12, y0 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 220, 80), 1, cv2.LINE_AA)

                self.last_pairs.append({
                    "lm": lm,
                    "cx": cx_n, "cy": cy_n,
                    "x_min": x_min, "x_max": x_max,
                    "y_min": y_min, "y_max": y_max,
                    "label": label
                })

        if self.last_pairs:
            cx_n = self.last_pairs[0]["cx"]
            cy_n = self.last_pairs[0]["cy"]
        else:
            cx_n, cy_n = 0.5, 0.0

        cx_n = _crop_map(cx_n, EDGE_MARGIN_X, EDGE_MARGIN_X)
        cy_n = _crop_map(cy_n, EDGE_MARGIN_Y, EDGE_MARGIN_Y)

        cx = cx_n * SCREEN_W
        cy = max(0.0, min(cy_n * SCREEN_H, SCREEN_H * 0.92))

        self.last_rgb = rgb

        self._median_x.append(cx)
        self._median_y.append(cy)
        mx = self._median(self._median_x)
        my = self._median(self._median_y)
        fx = self._euro_x.filter(mx, max(dt_frame, 1e-6))
        fy = self._euro_y.filter(my, max(dt_frame, 1e-6))

        ratio = None
        if self.last_pairs:
            lm0 = self.last_pairs[0]["lm"].landmark
            wrist, idx_tip, thm_tip, mid_mcp = lm0[0], lm0[8], lm0[4], lm0[9]
            base = math.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y) + 1e-6
            pinch = math.hypot(thm_tip.x - idx_tip.x, thm_tip.y - idx_tip.y)
            ratio = pinch / base

        if ratio is None:
            self._last_seen_t += dt_frame
            present = self._last_seen_t <= HOLD_MISS_T
        else:
            self._last_seen_t = 0.0
            if self._closed_state:
                self._closed_state = ratio < PINCH_OPEN
            else:
                self._closed_state = ratio < PINCH_CLOSE

        present = self._last_seen_t <= HOLD_MISS_T

        return ClawInput(fx, fy, self._closed_state, present)


    def camera_surface(self) -> Optional[pygame.Surface]:
        if self.last_rgb is None:
            return None
        h, w, _ = self.last_rgb.shape
        surf = pygame.image.frombuffer(self.last_rgb.tobytes(), (w, h), "RGB")
        if w != CAM_W or h != CAM_H:
            surf = pygame.transform.smoothscale(surf, (CAM_W, CAM_H))
        return surf

    def stop(self):
        self.hands.close()
        self.cap.release()


# --- PHYSICS ---
class PhysicsSim:
    def __init__(self):
        self.space = pymunk.Space()
        self.space.gravity = (0, GRAVITY_Y)
        self.space.iterations = SOLVER_ITER
        self.space.damping = GLOBAL_DAMPING
        self.space.sleep_time_threshold = SLEEP_THRESHOLD
        self.space.collision_slop = 0.2
        self.space.collision_bias = (1.0 - 0.15) ** 60

        self.blocks: List[pymunk.Shape] = []
        self._add_bounds()
        self._add_buckets()

        self.block_in_bucket: dict[pymunk.Shape, str | None] = {}
        self._flash_expire: dict[pymunk.Shape, int] = {}
        self._orig_color: dict[pymunk.Shape, tuple[int, int, int, int]] = {}

        self._despawn_at: dict[pymunk.Shape, int] = {}


    def _add_bounds(self):
        sb = self.space.static_body
        segs = [
            pymunk.Segment(sb, (PAD, FLOOR_Y), (SCREEN_W - PAD, FLOOR_Y), 6),
            pymunk.Segment(sb, (PAD, CEILING_Y), (PAD, FLOOR_Y), 6),
            pymunk.Segment(sb, (SCREEN_W - PAD, CEILING_Y), (SCREEN_W - PAD, FLOOR_Y), 6),
            pymunk.Segment(sb, (PAD, CEILING_Y), (SCREEN_W - PAD, CEILING_Y), 6),
        ]
        for s in segs:
            s.friction = 0.98
            s.elasticity = 0.0
            s.color = (90, 90, 100, 255)
        self.space.add(*segs)

    def _add_buckets(self):
        def make_bucket(x_left: int):
            sb = self.space.static_body
            x_right, yb, yt = x_left + BUCKET_W, FLOOR_Y, FLOOR_Y - BUCKET_H
            segs = [
                pymunk.Segment(sb, (x_left, yb), (x_left, yt), BUCKET_WALL),
                pymunk.Segment(sb, (x_right, yb), (x_right, yt), BUCKET_WALL),
                pymunk.Segment(sb, (x_left, yb), (x_right, yb), BUCKET_WALL),
            ]
            for s in segs:
                s.friction = 0.99
                s.elasticity = 0.0
                s.color = BUCKET_COLOR
            self.space.add(*segs)

        pad = 40
        make_bucket(pad)
        make_bucket(SCREEN_W - BUCKET_W - pad)

    def clear_all_dynamics(self):
        dyn_bodies = {b for b in self.space.bodies if b.body_type == pymunk.Body.DYNAMIC}
        for c in list(self.space.constraints):
            if c.a in dyn_bodies or c.b in dyn_bodies:
                self.space.remove(c)
        for b in list(dyn_bodies):
            for s in list(b.shapes):
                if s in self.space.shapes:
                    self.space.remove(s)
            if b in self.space.bodies:
                self.space.remove(b)
        self.blocks = [s for s in self.blocks if s.body.body_type != pymunk.Body.DYNAMIC]

    def remove_block(self, shp: pymunk.Shape):
        """Remove a single dynamic block and associated data from the space."""
        if shp not in self.blocks:
            return
        body = shp.body

        # Remove any constraints attached to this body
        for c in list(self.space.constraints):
            if c.a is body or c.b is body:
                self.space.remove(c)

        # Remove shape and body from the space
        if shp in self.space.shapes:
            try:
                self.space.remove(shp, body)
            except Exception:
                # fallback in case body already gone
                try:
                    self.space.remove(shp)
                except Exception:
                    pass

        # Bookkeeping
        if shp in self.blocks:
            self.blocks.remove(shp)
        self.block_in_bucket.pop(shp, None)
        self._flash_expire.pop(shp, None)
        self._orig_color.pop(shp, None)
        self._despawn_at.pop(shp, None)

    def add_box(self, x, y, w, h):
        mass = DENSITY * w * h
        moment = pymunk.moment_for_box(mass, (w, h))
        body = pymunk.Body(mass, moment)
        body.position = x, y
        shape = pymunk.Poly.create_box(body, (w, h))
        shape.friction = 0.9
        shape.elasticity = 0.0
        base = random.randint(0, 110)
        shape.color = (
            base,
            base,
            min(255, base + random.randint(20, 100)),
            255,
        )
        self.space.add(body, shape)
        self.blocks.append(shape)
        self.block_in_bucket[shape] = None
        self._orig_color[shape] = shape.color
        return shape

    def clear_boxes(self):
        target_bodies = {s.body for s in self.blocks}
        for c in list(self.space.constraints):
            if c.a in target_bodies or c.b in target_bodies:
                self.space.remove(c)
        for s in list(self.blocks):
            if s in self.space.shapes:
                self.space.remove(s, s.body)
        self.blocks.clear()


# --- CLAW ---
class Claw:
    """
    Kinematic follower with filtered input and soft attachment.
    Default mode holds one block.
    True-Collector mode can hold multiple True blocks simultaneously.
    """
    def __init__(self, space: pymunk.Space):
        self.space = space
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = (SCREEN_W * 0.5, 0.0)
        self.shape = pymunk.Circle(self.body, CLAW_RADIUS)
        self.shape.sensor = True
        self.space.add(self.body, self.shape)

        # single-hold fields for default mode
        self.holding_pivot: Optional[pymunk.PivotJoint] = None
        self.holding_spring: Optional[pymunk.DampedSpring] = None
        self.holding_shape: Optional[pymunk.Shape] = None

        # multi-hold for collector mode
        self.holding: list[tuple[pymunk.Shape, pymunk.PivotJoint, pymunk.DampedSpring]] = []
        self.max_grabs = COLLECT_MAX_GRABS
        self.collect_radius = COLLECT_RADIUS
        self._next_collect_ms = 0

        self.prev_pos = self.body.position
        self.prev_vel = pymunk.Vec2d(0.0, 0.0)

        self.collect_only_true = False
        self.state_closed = False

    def update(self, inp: ClawInput, dt: float):
        self.state_closed = bool(inp.present and inp.closed)
        new_pos = pymunk.Vec2d(inp.x, inp.y)
        raw_vel = (new_pos - self.prev_pos) / max(dt, 1e-6)

        dv = raw_vel - self.prev_vel
        max_dv = MAX_CLAW_ACC * dt
        if dv.length > max_dv:
            dv = dv.normalized() * max_dv
        vel = self.prev_vel + dv

        if vel.length > MAX_CLAW_SPEED:
            vel = vel.normalized() * MAX_CLAW_SPEED

        self.body.velocity = vel
        self.body.position = new_pos
        self.prev_pos = new_pos
        self.prev_vel = vel

        # single-hold path
        if not self.collect_only_true:
            if inp.present and inp.closed and not self.holding_pivot:
                self.try_grab()
            if (not inp.closed or not inp.present) and self.holding_pivot:
                self.release()
        else:
            if self.holding_pivot:
                self.release()

    def try_grab(self):
        pq = self.space.point_query_nearest(self.body.position, GRAB_MAX_DIST, pymunk.ShapeFilter())
        if not pq or pq.shape is None or pq.shape.body.body_type != pymunk.Body.DYNAMIC:
            return
        shp = pq.shape

        pivot = pymunk.PivotJoint(self.body, shp.body, self.body.position)
        pivot.max_force = GRAB_FORCE
        pivot.error_bias = (1.0 - 0.25) ** 60
        pivot.max_bias = 900.0

        spring = pymunk.DampedSpring(
            self.body, shp.body,
            (0, 0), (0, 0),
            rest_length=0.0,
            stiffness=SPRING_K,
            damping=SPRING_C,
        )
        self.space.add(pivot, spring)
        self.holding_pivot = pivot
        self.holding_spring = spring
        self.holding_shape = shp

    def release(self):
        if self.holding_pivot:
            try:
                self.space.remove(self.holding_pivot)
            except Exception:
                pass
        if self.holding_spring:
            try:
                self.space.remove(self.holding_spring)
            except Exception:
                pass
        self.holding_pivot = None
        self.holding_spring = None
        self.holding_shape = None

    # --- Collector mode API ---
    def collect_true(self, sim: "PhysicsSim", profile: dict, now_ms: int):
        if now_ms < self._next_collect_ms:
            return
        self._next_collect_ms = now_ms + COLLECT_COOLDOWN_MS
        if len(self.holding) >= self.max_grabs:
            return

        cx, cy = self.body.position
        # naive range scan over dynamic blocks
        for shp in sim.blocks:
            if len(self.holding) >= self.max_grabs:
                break
            if shp.body.body_type != pymunk.Body.DYNAMIC:
                continue
            if any(shp is s for s, _, _ in self.holding):
                continue
            dx = float(shp.body.position.x) - float(cx)
            dy = float(shp.body.position.y) - float(cy)
            if dx*dx + dy*dy > self.collect_radius * self.collect_radius:
                continue
            # True criterion
            if shp.body.mass < profile["avg_mass"]:
                continue
            pv = pymunk.PivotJoint(self.body, shp.body, self.body.position)
            pv.max_force = GRAB_FORCE
            pv.error_bias = (1.0 - 0.25) ** 60
            pv.max_bias = 900.0
            sp = pymunk.DampedSpring(self.body, shp.body, (0,0), (0,0), 0.0, SPRING_K, SPRING_C)
            self.space.add(pv, sp)
            self.holding.append((shp, pv, sp))

    def release_all(self):
        for shp, pv, sp in list(self.holding):
            try:
                self.space.remove(pv)
            except Exception:
                pass
            try:
                self.space.remove(sp)
            except Exception:
                pass
        self.holding.clear()

    def holding_any(self) -> bool:
        return bool(self.holding_pivot or self.holding)


def sorting_status(sim: "PhysicsSim", profile: dict, *, true_only: bool = False) -> tuple[bool, float, int, int, int]:
    """
    Returns: all_in, pct_correct, correct, total, in_buckets
    If true_only is True, only blocks with mass >= avg_mass are counted.
    """
    left, right = bucket_rects()

    blocks = sim.blocks if not true_only else [s for s in sim.blocks if s.body.mass >= profile["avg_mass"]]

    total = len(blocks)
    in_buckets = 0
    correct = 0

    for shp in blocks:
        x, y = shp.body.position
        side = None
        if left.collidepoint(int(x), int(y)):
            side = "T"
        elif right.collidepoint(int(x), int(y)):
            side = "F"
        else:
            continue  # not parked

        in_buckets += 1
        mass = shp.body.mass
        should = "F" if mass < profile["avg_mass"] else "T"
        if side == should:
            correct += 1

    all_in = (in_buckets == total) and total > 0
    pct = (correct / total * 100.0) if total > 0 else 0.0
    return all_in, pct, correct, total, in_buckets


# --- SPAWNING ---
def spawn_blocks(sim, n: int):
    wmin, wmax = W_RANGE
    hmin, hmax = H_RANGE
    for _ in range(n):
        w = random.uniform(wmin, wmax)
        h = random.uniform(hmin, hmax)
        x = random.uniform(BUCKET_W + w, SCREEN_W - BUCKET_W - w)
        y = random.uniform(-120, -20)
        sim.add_box(x, y, w, h)


# --- DRAWING ---
def _fmt_ms(ms: int) -> str:
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}.{ms:03d}"

def draw_menu(screen: pygame.Surface, font: pygame.font.Font, claw_pos: tuple[float, float], hover_idx: int | None, toggle_rect: pygame.Rect, collect_true_mode: bool):
    title_font = pygame.font.SysFont("consolas", 48)
    title = title_font.render("Select Difficulty", True, (220, 220, 240))
    screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 80))

    rects = menu_button_rects()
    for i, rect in enumerate(rects):
        name, count = DIFFICULTIES[i]
        pygame.draw.rect(screen, (40, 40, 52), rect, border_radius=8)
        pygame.draw.rect(screen, (90, 90, 110), rect, width=2, border_radius=8)
        if hover_idx == i:
            pygame.draw.rect(screen, (120, 160, 255), rect, width=3, border_radius=8)
        txt = font.render(f"{name}  ({count} blocks)", True, (230, 230, 235))
        screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    # toggle
    pygame.draw.rect(screen, (40, 40, 52), toggle_rect, border_radius=8)
    pygame.draw.rect(screen, (90, 90, 110), toggle_rect, width=2, border_radius=8)
    t = "Cerebra AI: ON" if collect_true_mode else "Cerebra AI: OFF"
    col = (180,255,180) if collect_true_mode else (230,230,235)
    t_surf = font.render(t, True, col)
    screen.blit(t_surf, (toggle_rect.centerx - t_surf.get_width() // 2, toggle_rect.centery - t_surf.get_height() // 2))

    # draw claw marker on menu
    cx, cy = claw_pos
    pygame.draw.circle(screen, (255, 210, 90), (int(cx), int(cy)), 10, 2)

def draw_buckets(screen: pygame.Surface, profile):
    font = pygame.font.SysFont("consolas", 96)
    left = pygame.Rect(40, FLOOR_Y - BUCKET_H, BUCKET_W, BUCKET_H)
    right = pygame.Rect(SCREEN_W - BUCKET_W - 40, FLOOR_Y - BUCKET_H, BUCKET_W, BUCKET_H)
    pygame.draw.rect(screen, (60, 60, 80), left, 2)
    pygame.draw.rect(screen, (60, 60, 80), right, 2)
    t_surf = font.render(f'T', True, (220, 220, 240))
    f_surf = font.render(f'F', True, (220, 220, 240))

    font = pygame.font.SysFont("consolas", 48)
    tn_surf = font.render(f"> {profile['avg_mass']:.2f} kg", True, (220, 220, 240))
    fn_surf = font.render(f"< {profile['avg_mass']:.2f} kg", True, (220, 220, 240))

    screen.blit(t_surf, (left.centerx - t_surf.get_width() // 2, left.top + 50))
    screen.blit(f_surf, (right.centerx - f_surf.get_width() // 2, right.top + 50))

    screen.blit(tn_surf, (left.centerx - tn_surf.get_width() // 2, left.top + 150))
    screen.blit(fn_surf, (right.centerx - fn_surf.get_width() // 2, right.top + 150))


def draw_claw(screen: pygame.Surface, claw: Claw):
    x, y = claw.body.position
    engaged = claw.holding_any() or claw.state_closed
    color = CLAW_COLOR_GRAB if engaged else CLAW_COLOR_IDLE
    pygame.draw.line(screen, (90, 90, 120), (int(x), 0), (int(x), int(y - CLAW_RADIUS)), 2)
    pygame.draw.circle(screen, color, (int(x), int(y)), CLAW_RADIUS, 2)

def draw_collect_count(screen: pygame.Surface, font: pygame.font.Font, claw: Claw):
    if not claw.holding:
        return
    bx, by = claw.body.position
    n = len(claw.holding)
    label = font.render(f"x{n}", True, (180, 255, 180))
    screen.blit(label, (int(bx) + 26, int(by) - 64))

def draw_weight(screen: pygame.Surface, font: pygame.font.Font, claw: Claw):
    if not claw.holding_shape:
        return
    shp = claw.holding_shape
    m = shp.body.mass
    bx, by = claw.body.position
    label = font.render(f"{m:.2f} kg", True, (255, 240, 150))
    screen.blit(label, (int(bx) + 26, int(by) - 40))


def draw_camera_panel(screen: pygame.Surface, tracker: HandTracker, pos: tuple[int, int]):
    x, y = pos
    rect = pygame.Rect(x - 4, y - 4, CAM_W + 8, CAM_H + 8)
    pygame.draw.rect(screen, (40, 40, 48), rect, border_radius=6)
    surf = tracker.camera_surface()
    if surf:
        screen.blit(surf, (x, y))
    pygame.draw.rect(screen, (80, 80, 96), rect, width=2, border_radius=6)

def bucket_rects() -> tuple[pygame.Rect, pygame.Rect]:
    left  = pygame.Rect(40, FLOOR_Y - BUCKET_H, BUCKET_W, BUCKET_H)
    right = pygame.Rect(SCREEN_W - BUCKET_W - 40, FLOOR_Y - BUCKET_H, BUCKET_W, BUCKET_H)
    return left, right

def reset_sim(sim: PhysicsSim, initial_count: int):
    sim.clear_boxes()
    profile = setup_mass_profile()
    print(
        f"[run] density={profile['density']:.6f}  avg≈{profile['avg_mass']:.2f} kg  max={profile['max_mass']:.2f} kg  "
        f"W∈{profile['w_range']} H∈{profile['h_range']}")
    spawn_blocks(sim, initial_count)
    return profile

def menu_update_selection(claw_input, last_select_time, now_time) -> tuple[int | None, int | None, float]:
    """
    Returns: hover_idx, chosen_idx, new_last_select_time
    chosen_idx is None when not selected this frame.
    """
    rects = menu_button_rects()
    hover_idx = None
    for i, r in enumerate(rects):
        if point_in_rect(claw_input.x, claw_input.y, r):
            hover_idx = i
            break
    chosen_idx = None
    if hover_idx is not None and claw_input.present and claw_input.closed:
        if now_time - last_select_time >= MENU_SELECT_COOLDOWN:
            chosen_idx = hover_idx
            last_select_time = now_time
    return hover_idx, chosen_idx, last_select_time



def main():
    # --- minimal wave spawner (local to main) ---
    class WaveSpawner:
        """
        Spawns total_count blocks in bursts of batch_size every interval_ms.
        Call .update(now_ms, sim) each frame.
        """
        def __init__(self, total_count: int, batch_size: int = 4, interval_ms: int = 120):
            self.total = int(total_count)
            self.batch = int(batch_size)
            self.interval = int(interval_ms)
            self.spawned = 0
            self.next_ms = 0
            self.done = (self.total <= 0)

        def start(self, now_ms: int):
            self.spawned = 0
            self.next_ms = now_ms
            self.done = (self.total <= 0)

        def update(self, now_ms: int, sim: "PhysicsSim") -> int:
            if self.done or now_ms < self.next_ms:
                return 0
            n = min(self.batch, self.total - self.spawned)
            if n > 0:
                spawn_blocks(sim, n)
                self.spawned += n
            self.next_ms = now_ms + self.interval
            if self.spawned >= self.total:
                self.done = True
            return n

    # --- init ---
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Robotics Demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)
    score_font = pygame.font.SysFont("consolas", 28)

    sim = PhysicsSim()
    claw = Claw(sim.space)
    tracker = HandTracker(CAM_IDX)
    draw_opts = pymunk.pygame_util.DrawOptions(screen)

    # --- game state ---
    STATE_MENU, STATE_PLAY = 0, 1
    DIFFICULTIES = [("Easy", 8), ("Medium", 16), ("Hard", 28)]

    state = STATE_MENU
    menu_last_select_t = 0.0
    current_difficulty_name = None
    current_initial_count = None

    # True-Collector toggle
    collect_true_mode = False

    # waves
    active_waves: list[WaveSpawner] = []

    # mass profile
    profile = setup_mass_profile()

    # timer
    timer_running = False
    timer_start_ms = 0
    final_time_ms = None

    # physics stepping
    acc = 0.0

    # precompute toggle rect under the difficulty buttons
    rects = menu_button_rects()
    lowest = rects[-1]
    toggle_rect = pygame.Rect(
        (SCREEN_W - TOGGLE_W) // 2,
        lowest.bottom + 2 * MENU_BTN_GAP,
        TOGGLE_W,
        TOGGLE_H
    )

    running = True
    while running:
        dt_frame = clock.tick(FPS_CAP) / 1000.0
        acc += dt_frame
        now_ms = pygame.time.get_ticks()
        now_sec = now_ms / 1000.0

        # events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False

                elif e.key == pygame.K_m:
                    claw.release()
                    claw.release_all()
                    sim.clear_all_dynamics()
                    active_waves.clear()
                    state = STATE_MENU
                    current_difficulty_name = None
                    current_initial_count = None
                    timer_running = False
                    final_time_ms = None

                elif state == STATE_PLAY and e.key == pygame.K_SPACE:
                    if current_initial_count is None:
                        current_initial_count = 8
                    w = WaveSpawner(total_count=current_initial_count, batch_size=4, interval_ms=120)
                    w.start(now_ms)
                    active_waves.append(w)

                elif state == STATE_PLAY and e.key == pygame.K_c:
                    claw.release()
                    claw.release_all()
                    sim.clear_all_dynamics()
                    active_waves.clear()

                elif state == STATE_PLAY and e.key == pygame.K_r:
                    if current_initial_count is None:
                        current_initial_count = 8
                    claw.release()
                    claw.release_all()
                    sim.clear_all_dynamics()
                    profile = setup_mass_profile()
                    active_waves.clear()
                    w = WaveSpawner(total_count=current_initial_count, batch_size=4, interval_ms=120)
                    w.start(now_ms)
                    active_waves.append(w)
                    timer_running = True
                    timer_start_ms = now_ms
                    final_time_ms = None

        # hand input
        inp = tracker.read(dt_frame)

        # ----------------------
        # STATE: MENU
        # ----------------------
        if state == STATE_MENU:
            claw.prev_pos = claw.body.position
            claw.body.position = (inp.x, inp.y)
            claw.prev_vel = pymunk.Vec2d(0, 0)

            # toggle interaction
            if inp.present and inp.closed and point_in_rect(inp.x, inp.y, toggle_rect):
                if now_sec - menu_last_select_t >= MENU_SELECT_COOLDOWN:
                    collect_true_mode = not collect_true_mode
                    SND_CLICK.play()
                    menu_last_select_t = now_sec


            # difficulty selection
            hover_idx, chosen_idx, menu_last_select_t = menu_update_selection(inp, menu_last_select_t, now_sec)
            claw.collect_only_true = collect_true_mode

            # draw menu scene
            screen.fill(BG_COLOR)
            draw_camera_panel(screen, tracker, (12, 12))
            draw_menu(screen, font, claw.body.position, hover_idx, toggle_rect, collect_true_mode)
            pygame.display.flip()

            # on pick: set difficulty, reset run, and start initial wave
            if chosen_idx is not None:
                SND_CLICK.play()
                name, initial_count = DIFFICULTIES[chosen_idx]
                current_difficulty_name = name
                current_initial_count = initial_count

                claw.release()
                claw.release_all()
                sim.clear_all_dynamics()
                profile = setup_mass_profile()

                active_waves.clear()
                w = WaveSpawner(total_count=current_initial_count, batch_size=4, interval_ms=120)
                w.start(now_ms)
                active_waves.append(w)

                timer_running = True
                timer_start_ms = now_ms
                final_time_ms = None


                state = STATE_PLAY
            continue  # next frame

        # ----------------------
        # STATE: PLAY
        # ----------------------

        if hasattr(sim, "update_spawn_grace"):
            sim.update_spawn_grace(now_ms)

        if active_waves:
            for w in list(active_waves):
                w.update(now_ms, sim)
                if w.done:
                    active_waves.remove(w)

        # update claw kinematics
        claw.update(inp, max(dt_frame, 1e-6))

        # collector vacuuming when closed
        if collect_true_mode and inp.present and inp.closed:
            claw.collect_true(sim, profile, now_ms)

        # physics fixed step
        while acc >= DT_FIXED:
            sim.space.step(DT_FIXED)
            acc -= DT_FIXED

        cx = float(claw.body.position.x)

        global CAM_LEFT
        if CAM_LEFT:
            if cx <= CAM_PAD + CAM_W + FLIP_HYSTERESIS:
                CAM_LEFT = False
        else:
            if cx >= SCREEN_W - CAM_PAD - CAM_W - FLIP_HYSTERESIS:
                CAM_LEFT = True

        cam_x = CAM_PAD if CAM_LEFT else (SCREEN_W - CAM_PAD - CAM_W)
        cam_y = CAM_PAD

        update_color_flashes(sim)

        # --- timed despawn of correctly placed blocks ---
        now = pygame.time.get_ticks()
        for shp, t_exp in list(sim._despawn_at.items()):
            if now >= t_exp:
                sim.remove_block(shp)


        left, right = bucket_rects()

        # auto-dump held True blocks when claw is over the True bucket
        if collect_true_mode and claw.holding:
            bx, by = claw.body.position
            if left.collidepoint(int(bx), int(by)):
                claw.release_all()

        for shp in list(sim.blocks):
            x, y = shp.body.position
            prev_state = sim.block_in_bucket.get(shp)

            new_state = None
            if left.collidepoint(int(x), int(y)):
                new_state = "T"
            elif right.collidepoint(int(x), int(y)):
                new_state = "F"

            if new_state and new_state != prev_state:
                sim.block_in_bucket[shp] = new_state
                mass = shp.body.mass
                should = "F" if mass < profile["avg_mass"] else "T"
                if new_state == should:
                    trigger_color_flash(sim, shp, (60, 250, 90))  # GREEN fill
                    SND_CORRECT.play()
                    # NEW: schedule despawn shortly after the flash
                    sim._despawn_at[shp] = pygame.time.get_ticks() + 1000

                else:
                    trigger_color_flash(sim, shp, (230, 60, 60))
                    SND_INCORRECT.play()

            elif new_state is None and prev_state is not None:
                sim.block_in_bucket[shp] = None

        # scoring and timer stop condition
        all_in, pct, correct, total, in_buckets = sorting_status(sim, profile, true_only=collect_true_mode)
        if timer_running and all_in:
            final_time_ms = pygame.time.get_ticks() - timer_start_ms
            timer_running = False

        # draw play scene
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, (28, 28, 32), (0, FLOOR_Y, SCREEN_W, SCREEN_H - FLOOR_Y))
        sim.space.debug_draw(draw_opts)
        draw_buckets(screen, profile)
        draw_claw(screen, claw)
        draw_collect_count(screen, font, claw)
        draw_weight(screen, font, claw)
        draw_camera_panel(screen, tracker, (cam_x, cam_y))

        # banner
        if collect_true_mode:
            line = (f"True placed: {pct:.0f}%  ({correct}/{total})" if all_in
                    else f"True in buckets: {in_buckets}/{total}")
        else:
            line = (f"Sorted correctly: {pct:.0f}%  ({correct}/{total})" if all_in
                    else f"In buckets: {in_buckets}/{total}")
        score_col = (190, 255, 190) if all_in else (180, 180, 190)
        score_surf = score_font.render(line, True, score_col)
        score_x = SCREEN_W // 2 - score_surf.get_width() // 2
        score_y = 16
        screen.blit(score_surf, (score_x, score_y))

        # timer just below the banner
        elapsed_ms = (final_time_ms if final_time_ms is not None else (now_ms - timer_start_ms)) if (timer_running or final_time_ms is not None) else 0
        timer_text = f"time { _fmt_ms(int(elapsed_ms)) }"
        timer_surf = font.render(timer_text, True, (200, 200, 210))
        timer_x = SCREEN_W // 2 - timer_surf.get_width() // 2
        timer_y = score_y + score_surf.get_height() + 6
        screen.blit(timer_surf, (timer_x, timer_y))

        # fps HUD
        fps_txt = font.render(f"fps {clock.get_fps():5.1f}", True, (200, 200, 210))
        screen.blit(fps_txt, (12, SCREEN_H - 28))

        pygame.display.flip()

    # shutdown
    tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
