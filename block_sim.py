# --- IMPORTS ---
import math
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


# --- CONFIG ---
SCREEN_W = 1920
SCREEN_H = 1080
BG_COLOR = (18, 18, 20)
FLOOR_Y = SCREEN_H - 60

GRAVITY_Y = 1600.0

DENSITY = 0.0008
W_RANGE = (40, 100)
H_RANGE = (40, 100)

FPS_CAP = 120                  # display cap
SOLVER_ITER = 60            # was 40
DT_FIXED = 1.0 / 300.0      # was 1/240         # more iterations
GLOBAL_DAMPING = 0.99
SLEEP_THRESHOLD = 0.3

BUCKET_W = SCREEN_W // 6
BUCKET_H = SCREEN_H // 2
BUCKET_WALL = 8
BUCKET_COLOR = (120, 120, 140, 255)

CLAW_RADIUS = 18
CLAW_COLOR_IDLE = (180, 200, 255)
CLAW_COLOR_GRAB = (255, 200, 60)

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17)              # palm crossbars (optional)
]



# Grabbing
GRAB_MAX_DIST = 28
GRAB_FORCE = 2.0e5          # was 7.5e4
SPRING_K  = 8000.0          # was 2000
SPRING_C  = 1200.0          # was 450

# Camera panel
CAM_MIRROR = True
CAM_W = 480
CAM_H = 270
CAM_POS = (12, 12)

CAM_PAD = 12
FLIP_HYSTERESIS = 24  # pixels to prevent flicker

PAD = 40
CEILING_Y = -200

CAM_LEFT = True  # True = top-left, False = top-right

# Input cropping: percent of camera frame ignored on each side
EDGE_MARGIN_X = 0.15   # 0.10–0.20 works well
EDGE_MARGIN_Y = 0.10

# Hand filtering and classification
MAX_CLAW_SPEED = 3000.0     # was 2200
MAX_CLAW_ACC   = 90000.0    # was 42000        # px/s^2 cap
PINCH_CLOSE = 0.55              # hysteresis low
PINCH_OPEN = 0.65               # hysteresis high
HOLD_MISS_T = 0.08              # seconds to hold last valid hand
MEDIAN_WIN = 5                  # frames
OE_MINCUTOFF = .8               # One Euro position min cutoff
OE_BETA = 0.05                  # One Euro speed coefficient
OE_DCUTOFF = 1.0                # One Euro derivative cutoff

# --- Game states ---
STATE_MENU = 0
STATE_PLAY = 1

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
MENU_SELECT_COOLDOWN = 0.40  # seconds debounce after a pinch-select


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

    # Randomize size ranges each run
    wmin = random.randint(36, 64)
    wmax = random.randint(96, 140)
    hmin = random.randint(36, 64)
    hmax = random.randint(96, 140)
    W_RANGE = (wmin, wmax)
    H_RANGE = (hmin, hmax)

    # Choose a target max mass for this run (kg)
    M_max_run = random.uniform(6.0, 18.0)

    # Derive density so the largest possible block hits that max mass
    A_max = wmax * hmax
    DENSITY = M_max_run / A_max  # kg per px^2

    # Compute implied average for HUD
    Ew = 0.5 * (wmin + wmax)
    Eh = 0.5 * (hmin + hmax)
    E_area = Ew * Eh
    E_mass = DENSITY * E_area
    return {
        "w_range": W_RANGE, "h_range": H_RANGE,
        "max_mass": M_max_run, "avg_mass": E_mass,
        "density": DENSITY
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
        return 0.5  # degenerate safety
    n = (n - lo) / span
    return 0.0 if n < 0.0 else 1.0 if n > 1.0 else n


class HandTracker:
    """
    MediaPipe hands -> screen coords + open/closed.
    Jitter reduction: median over last N and One Euro filtering.
    Hysteresis on pinch classification. Optional mirror.
    """
    def __init__(self, cam_index: int = 0):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Camera not found or cannot be opened.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
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
        self._last_seen_t = 0.0  # seconds since last valid landmarks

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
                    label = hds[idx].classification[0].label  # 'Left' or 'Right'

                xs = np.fromiter((p.x for p in lm.landmark), dtype=np.float32, count=21)
                ys = np.fromiter((p.y for p in lm.landmark), dtype=np.float32, count=21)

                x_min, x_max = float(xs.min()), float(xs.max())
                y_min, y_max = float(ys.min()), float(ys.max())
                cx_n = 0.5 * (x_min + x_max)
                cy_n = 0.5 * (y_min + y_max)

                # Draw bones
                for a, b in HAND_CONN:
                    ax, ay = int(xs[a] * w), int(ys[a] * h)
                    bx, by = int(xs[b] * w), int(ys[b] * h)
                    cv2.line(rgb, (ax, ay), (bx, by), (80, 220, 255), 1, cv2.LINE_AA)

                # Draw joints
                for i in range(21):
                    jx, jy = int(xs[i] * w), int(ys[i] * h)
                    cv2.circle(rgb, (jx, jy), 2, (255, 230, 80), -1, cv2.LINE_AA)

                # Bbox and label
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

        # Choose control point (center of hand box; swap to fingertip if preferred)
        if self.last_pairs:
            cx_n = self.last_pairs[0]["cx"]
            cy_n = self.last_pairs[0]["cy"]
        else:
            cx_n, cy_n = 0.5, 0.0

        cx_n = _crop_map(cx_n, EDGE_MARGIN_X, EDGE_MARGIN_X)
        cy_n = _crop_map(cy_n, EDGE_MARGIN_Y, EDGE_MARGIN_Y)

        # Convert to screen coordinates
        cx = cx_n * SCREEN_W
        cy = max(0.0, min(cy_n * SCREEN_H, SCREEN_H * 0.92))

        # Store overlay for the camera panel
        self.last_rgb = rgb

        # Filtering
        self._median_x.append(cx)
        self._median_y.append(cy)
        mx = self._median(self._median_x)
        my = self._median(self._median_y)
        fx = self._euro_x.filter(mx, max(dt_frame, 1e-6))
        fy = self._euro_y.filter(my, max(dt_frame, 1e-6))

        # Pinch ratio for open/closed using landmarks, no drawing utils
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

    def _add_bounds(self):
        sb = self.space.static_body
        segs = [
            # floor
            pymunk.Segment(sb, (PAD, FLOOR_Y), (SCREEN_W - PAD, FLOOR_Y), 6),
            # left wall (extend to ceiling)
            pymunk.Segment(sb, (PAD, CEILING_Y), (PAD, FLOOR_Y), 6),
            # right wall (extend to ceiling)
            pymunk.Segment(sb, (SCREEN_W - PAD, CEILING_Y), (SCREEN_W - PAD, FLOOR_Y), 6),
            # ceiling (off-screen)
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
        """
        Remove every DYNAMIC body (and their shapes/constraints) from the space.
        Keeps static geometry (floor, walls, buckets) intact.
        Also clears self.blocks to avoid stale references.
        """
        # 1) Remove constraints attached to any dynamic body
        dyn_bodies = {b for b in self.space.bodies if b.body_type == pymunk.Body.DYNAMIC}
        for c in list(self.space.constraints):
            if c.a in dyn_bodies or c.b in dyn_bodies:
                self.space.remove(c)

        # 2) Remove shapes belonging to dynamic bodies
        for b in list(dyn_bodies):
            for s in list(b.shapes):
                if s in self.space.shapes:
                    self.space.remove(s)
            if b in self.space.bodies:
                self.space.remove(b)

        # 3) Reset our bookkeeping
        self.blocks = [s for s in self.blocks if s.body.body_type != pymunk.Body.DYNAMIC]

    def add_box(self, x, y, w, h):
        mass = DENSITY * w * h
        moment = pymunk.moment_for_box(mass, (w, h))
        body = pymunk.Body(mass, moment)
        body.position = x, y
        shape = pymunk.Poly.create_box(body, (w, h))
        shape.friction = 0.9
        shape.elasticity = 0.0
        shape.color = (
            random.randint(110, 230),
            random.randint(110, 230),
            random.randint(110, 230),
            255,
        )
        self.space.add(body, shape)
        self.blocks.append(shape)
        return shape

    def clear_boxes(self):
        # bodies to delete
        target_bodies = {s.body for s in self.blocks}
        # remove constraints touching those bodies
        for c in list(self.space.constraints):
            if c.a in target_bodies or c.b in target_bodies:
                self.space.remove(c)
        # remove shapes and bodies
        for s in list(self.blocks):
            if s in self.space.shapes:
                self.space.remove(s, s.body)
        self.blocks.clear()


# --- CLAW ---
class Claw:
    """
    Kinematic follower with filtered input and soft attachment.
    Uses PivotJoint + DampedSpring to avoid chatter.
    """
    def __init__(self, space: pymunk.Space):
        self.space = space
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.body.position = (SCREEN_W * 0.5, 0.0)
        self.shape = pymunk.Circle(self.body, CLAW_RADIUS)
        self.shape.sensor = True
        self.space.add(self.body, self.shape)

        self.holding_pivot: Optional[pymunk.PivotJoint] = None
        self.holding_spring: Optional[pymunk.DampedSpring] = None
        self.holding_shape: Optional[pymunk.Shape] = None
        self.prev_pos = self.body.position
        self.prev_vel = pymunk.Vec2d(0.0, 0.0)

        self.state_closed = False

    def update(self, inp: ClawInput, dt: float):
        self.state_closed = bool(inp.present and inp.closed)
        # compute desired velocity with caps
        new_pos = pymunk.Vec2d(inp.x, inp.y)
        raw_vel = (new_pos - self.prev_pos) / max(dt, 1e-6)

        # acceleration cap
        dv = raw_vel - self.prev_vel
        max_dv = MAX_CLAW_ACC * dt
        if dv.length > max_dv:
            dv = dv.normalized() * max_dv
        vel = self.prev_vel + dv

        # speed cap
        if vel.length > MAX_CLAW_SPEED:
            vel = vel.normalized() * MAX_CLAW_SPEED

        self.body.velocity = vel
        self.body.position = new_pos
        self.prev_pos = new_pos
        self.prev_vel = vel

        if inp.present and inp.closed and not self.holding_pivot:
            self.try_grab()
        if (not inp.closed or not inp.present) and self.holding_pivot:
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

        # soft tether to damp oscillations
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


def sorting_status(sim: PhysicsSim, profile) -> tuple[bool, float, int, int, int]:
    """
    Returns: all_in, pct_correct, correct, total, in_buckets
    """
    left, right = bucket_rects()
    total = len(sim.blocks)
    in_buckets = 0
    correct = 0

    for shp in sim.blocks:
        x, y = shp.body.position
        side = None
        if left.collidepoint(int(x), int(y)):
            side = "T"
        elif right.collidepoint(int(x), int(y)):
            side = "F"
        else:
            continue  # not parked in a bucket yet

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

def draw_menu(screen: pygame.Surface, font: pygame.font.Font, claw_pos: tuple[float, float], hover_idx: int | None):
    title_font = pygame.font.SysFont("consolas", 48)
    title = title_font.render("Select Difficulty", True, (220, 220, 240))
    screen.blit(title, (SCREEN_W // 2 - title.get_width() // 2, 80))

    rects = menu_button_rects()
    for i, rect in enumerate(rects):
        name, count = DIFFICULTIES[i]
        # base
        pygame.draw.rect(screen, (40, 40, 52), rect, border_radius=8)
        pygame.draw.rect(screen, (90, 90, 110), rect, width=2, border_radius=8)
        # hover
        if hover_idx == i:
            pygame.draw.rect(screen, (120, 160, 255), rect, width=3, border_radius=8)
        # label
        txt = font.render(f"{name}  ({count} blocks)", True, (230, 230, 235))
        screen.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

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
    engaged = claw.holding_pivot or claw.state_closed
    color = CLAW_COLOR_GRAB if engaged else CLAW_COLOR_IDLE
    pygame.draw.line(screen, (90, 90, 120), (int(x), 0), (int(x), int(y - CLAW_RADIUS)), 2)
    pygame.draw.circle(screen, color, (int(x), int(y)), CLAW_RADIUS, 2)


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
        Call .update(now_ms, sim) each frame; it returns how many were spawned.
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
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Blocks + Buckets + Camera Claw")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 22)
    score_font = pygame.font.SysFont("consolas", 28)

    sim = PhysicsSim()
    claw = Claw(sim.space)
    tracker = HandTracker()
    draw_opts = pymunk.pygame_util.DrawOptions(screen)

    # --- game state ---
    STATE_MENU, STATE_PLAY = 0, 1
    DIFFICULTIES = [("Easy", 8), ("Medium", 16), ("Hard", 28)]

    state = STATE_MENU
    menu_last_select_t = 0.0  # pinch-select debounce
    current_difficulty_name = None
    current_initial_count = None

    # waves
    active_waves: list[WaveSpawner] = []

    # mass profile (replaced on selection/reset)
    profile = setup_mass_profile()

    # --- timer ---
    timer_running = False
    timer_start_ms = 0
    final_time_ms = None

    # --- physics stepping ---
    acc = 0.0  # fixed-step accumulator

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

                # back to menu (for testing)
                elif e.key == pygame.K_m:
                    claw.release()
                    sim.clear_all_dynamics()
                    active_waves.clear()
                    state = STATE_MENU
                    current_difficulty_name = None
                    current_initial_count = None
                    timer_running = False
                    final_time_ms = None

                # gameplay-only keys
                elif state == STATE_PLAY and e.key == pygame.K_SPACE:
                    # add an extra wave equal to the difficulty’s initial count
                    if current_initial_count is None:
                        current_initial_count = 8
                    w = WaveSpawner(total_count=current_initial_count, batch_size=4, interval_ms=120)
                    w.start(now_ms)
                    active_waves.append(w)

                elif state == STATE_PLAY and e.key == pygame.K_c:
                    # clear all blocks and cancel pending waves
                    claw.release()
                    sim.clear_all_dynamics()
                    active_waves.clear()

                elif state == STATE_PLAY and e.key == pygame.K_r:
                    # reset mass profile and spawn a fresh wave for current difficulty
                    if current_initial_count is None:
                        current_initial_count = 8
                    claw.release()
                    sim.clear_all_dynamics()
                    profile = setup_mass_profile()
                    active_waves.clear()
                    w = WaveSpawner(total_count=current_initial_count, batch_size=4, interval_ms=120)
                    w.start(now_ms)
                    active_waves.append(w)
                    # reset timer
                    timer_running = True
                    timer_start_ms = now_ms
                    final_time_ms = None

        # hand input
        inp = tracker.read(dt_frame)

        # ----------------------
        # STATE: MENU
        # ----------------------
        if state == STATE_MENU:
            # move visual claw cursor (no physics in menu)
            claw.prev_pos = claw.body.position
            claw.body.position = (inp.x, inp.y)
            claw.prev_vel = pymunk.Vec2d(0, 0)

            # selection via pinch
            hover_idx, chosen_idx, menu_last_select_t = menu_update_selection(inp, menu_last_select_t, now_sec)

            # draw menu scene
            screen.fill(BG_COLOR)
            draw_camera_panel(screen, tracker, (12, 12))   # fixed corner to avoid overlap
            draw_menu(screen, font, claw.body.position, hover_idx)
            pygame.display.flip()

            # on pick: set difficulty, reset run, and start initial wave
            if chosen_idx is not None:
                name, initial_count = DIFFICULTIES[chosen_idx]
                current_difficulty_name = name
                current_initial_count = initial_count

                claw.release()
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

        # optional: update temporary spawn grace if implemented
        if hasattr(sim, "update_spawn_grace"):
            sim.update_spawn_grace(now_ms)

        # advance all active waves; remove finished ones
        if active_waves:
            for w in list(active_waves):
                w.update(now_ms, sim)
                if w.done:
                    active_waves.remove(w)

        # claw + physics
        claw.update(inp, max(dt_frame, 1e-6))
        while acc >= DT_FIXED:
            sim.space.step(DT_FIXED)
            acc -= DT_FIXED

        # scoring and timer stop condition
        all_in, pct, correct, total, in_buckets = sorting_status(sim, profile)
        if timer_running and all_in:
            final_time_ms = pygame.time.get_ticks() - timer_start_ms
            timer_running = False

        # draw play scene
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, (28, 28, 32), (0, FLOOR_Y, SCREEN_W, SCREEN_H - FLOOR_Y))
        sim.space.debug_draw(draw_opts)
        draw_buckets(screen, profile)         # if your draw_buckets expects profile
        draw_claw(screen, claw)
        draw_weight(screen, font, claw)
        draw_camera_panel(screen, tracker, (12, 12))  # or your dynamic placement

        # banner: progress or final score
        line = (f"Sorted correctly: {pct:.0f}%  ({correct}/{total})" if all_in
                else f"In buckets: {in_buckets}/{total}")
        score_col = (190, 255, 190) if all_in else (180, 180, 190)
        score_surf = score_font.render(line, True, score_col)
        score_x = SCREEN_W // 2 - score_surf.get_width() // 2
        score_y = 16
        screen.blit(score_surf, (score_x, score_y))

        # timer just below the score/progress
        elapsed_ms = (final_time_ms if final_time_ms is not None else (now_ms - timer_start_ms)) if (timer_running or final_time_ms is not None) else 0
        timer_text = f"time { _fmt_ms(int(elapsed_ms)) }"
        timer_surf = font.render(timer_text, True, (200, 200, 210))
        timer_x = SCREEN_W // 2 - timer_surf.get_width() // 2
        timer_y = score_y + score_surf.get_height() + 6
        screen.blit(timer_surf, (timer_x, timer_y))

        # fps HUD (optional)
        fps_txt = font.render(f"fps {clock.get_fps():5.1f}", True, (200, 200, 210))
        screen.blit(fps_txt, (12, SCREEN_H - 28))

        pygame.display.flip()

    # shutdown
    tracker.stop()
    pygame.quit()


