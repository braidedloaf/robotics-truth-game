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


# --- CONFIG ---
SCREEN_W = 1920
SCREEN_H = 1080
BG_COLOR = (18, 18, 20)
FLOOR_Y = SCREEN_H - 60

GRAVITY_Y = 1600.0
DENSITY = 0.0008

FPS_CAP = 120                  # display cap
DT_FIXED = 1.0 / 240.0         # smaller physics step
SOLVER_ITER = 40               # more iterations
GLOBAL_DAMPING = 0.99
SLEEP_THRESHOLD = 0.3

SPAWN_BURST = 7

BUCKET_W = SCREEN_W // 6
BUCKET_H = SCREEN_H // 2
BUCKET_WALL = 8
BUCKET_COLOR = (120, 120, 140, 255)

CLAW_RADIUS = 18
CLAW_COLOR_IDLE = (180, 200, 255)
CLAW_COLOR_GRAB = (255, 200, 60)

# Grabbing
GRAB_MAX_DIST = 28
GRAB_FORCE = 7.5e4
SPRING_K = 2.0e3                # translational stiffness
SPRING_C = 450.0                # damping

# Camera panel
CAM_MIRROR = True
CAM_W = 480
CAM_H = 270
CAM_POS = (12, 12)

# Hand filtering and classification
MAX_CLAW_SPEED = 2200.0         # px/s cap
MAX_CLAW_ACC = 42000.0          # px/s^2 cap
PINCH_CLOSE = 0.42              # hysteresis low
PINCH_OPEN = 0.50               # hysteresis high
HOLD_MISS_T = 0.08              # seconds to hold last valid hand
MEDIAN_WIN = 5                  # frames
OE_MINCUTOFF = 1.2              # One Euro position min cutoff
OE_BETA = 0.02                  # One Euro speed coefficient
OE_DCUTOFF = 1.0                # One Euro derivative cutoff


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
            self._last_seen_t += dt_frame
            present = self._last_seen_t <= HOLD_MISS_T
            # Hold previous filtered position if within hold window
            x = self._euro_x._x_prev if self._euro_x._x_prev is not None else SCREEN_W / 2
            y = self._euro_y._x_prev if self._euro_y._x_prev is not None else 0.0
            return ClawInput(float(x), float(y), self._closed_state, present)

        if CAM_MIRROR:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_rgb = rgb

        res = self.hands.process(rgb)
        if not res.multi_hand_landmarks:
            self._last_seen_t += dt_frame
            present = self._last_seen_t <= HOLD_MISS_T
            x = self._euro_x._x_prev if self._euro_x._x_prev is not None else SCREEN_W / 2
            y = self._euro_y._x_prev if self._euro_y._x_prev is not None else 0.0
            return ClawInput(float(x), float(y), self._closed_state, present)

        self._last_seen_t = 0.0
        lm = res.multi_hand_landmarks[0].landmark
        wrist, idx_tip, thm_tip, mid_mcp = lm[0], lm[8], lm[4], lm[9]

        cx = idx_tip.x * SCREEN_W
        cy = idx_tip.y * SCREEN_H
        cy = max(0.0, min(cy, SCREEN_H * 0.92))

        # median pre-filter
        self._median_x.append(cx)
        self._median_y.append(cy)
        mx = self._median(self._median_x)
        my = self._median(self._median_y)

        # one euro filtering
        fx = self._euro_x.filter(mx, max(dt_frame, 1e-6))
        fy = self._euro_y.filter(my, max(dt_frame, 1e-6))

        # hysteresis pinch
        base = math.hypot(mid_mcp.x - wrist.x, mid_mcp.y - wrist.y) + 1e-6
        pinch = math.hypot(thm_tip.x - idx_tip.x, thm_tip.y - idx_tip.y)
        ratio = pinch / base
        if self._closed_state:
            self._closed_state = ratio < PINCH_OPEN
        else:
            self._closed_state = ratio < PINCH_CLOSE

        return ClawInput(fx, fy, self._closed_state, True)

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
            pymunk.Segment(sb, (0, FLOOR_Y), (SCREEN_W, FLOOR_Y), 6),
            pymunk.Segment(sb, (0, 0), (0, FLOOR_Y), 6),
            pymunk.Segment(sb, (SCREEN_W, 0), (SCREEN_W, FLOOR_Y), 6),
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

    def add_box(self, x, y, w, h):
        mass = max(1.0, DENSITY * w * h)
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

    def update(self, inp: ClawInput, dt: float):
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


# --- SPAWNING ---
def spawn_blocks(sim: PhysicsSim, n: int):
    for _ in range(n):
        w, h = random.randint(40, 100), random.randint(40, 100)
        x = random.uniform(120, SCREEN_W - 120)
        y = random.uniform(-120, -20)
        sim.add_box(x, y, w, h)


# --- DRAWING ---
def draw_buckets(screen: pygame.Surface):
    font = pygame.font.SysFont("consolas", 96)
    left = pygame.Rect(40, FLOOR_Y - BUCKET_H, BUCKET_W, BUCKET_H)
    right = pygame.Rect(SCREEN_W - BUCKET_W - 40, FLOOR_Y - BUCKET_H, BUCKET_W, BUCKET_H)
    pygame.draw.rect(screen, (60, 60, 80), left, 2)
    pygame.draw.rect(screen, (60, 60, 80), right, 2)
    t_surf = font.render("T", True, (220, 220, 240))
    f_surf = font.render("F", True, (220, 220, 240))
    screen.blit(t_surf, (left.centerx - t_surf.get_width() // 2, left.top + 50))
    screen.blit(f_surf, (right.centerx - f_surf.get_width() // 2, right.top + 50))


def draw_claw(screen: pygame.Surface, claw: Claw):
    x, y = claw.body.position
    color = CLAW_COLOR_GRAB if claw.holding_pivot else CLAW_COLOR_IDLE
    pygame.draw.line(screen, (90, 90, 120), (int(x), 0), (int(x), int(y - CLAW_RADIUS)), 2)
    pygame.draw.circle(screen, color, (int(x), int(y)), CLAW_RADIUS, 2)


def draw_weight(screen: pygame.Surface, font: pygame.font.Font, claw: Claw):
    if not claw.holding_shape:
        return
    shp = claw.holding_shape
    m = shp.body.mass
    bx, by = shp.body.position
    label = font.render(f"{m:.1f} kg", True, (255, 240, 150))
    screen.blit(label, (int(bx) + 14, int(by) - 26))


def draw_camera_panel(screen: pygame.Surface, tracker: HandTracker):
    surf = tracker.camera_surface()
    rect = pygame.Rect(CAM_POS[0] - 4, CAM_POS[1] - 4, CAM_W + 8, CAM_H + 8)
    pygame.draw.rect(screen, (40, 40, 48), rect, border_radius=6)
    if surf:
        screen.blit(surf, CAM_POS)
    pygame.draw.rect(screen, (80, 80, 96), rect, width=2, border_radius=6)


# --- MAIN ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Blocks + Buckets + Camera Claw")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    sim = PhysicsSim()
    claw = Claw(sim.space)
    tracker = HandTracker()
    draw_opts = pymunk.pygame_util.DrawOptions(screen)

    spawn_blocks(sim, 14)

    running = True
    acc = 0.0
    while running:
        dt_frame = clock.tick(FPS_CAP) / 1000.0
        acc += dt_frame

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    spawn_blocks(sim, SPAWN_BURST)

        inp = tracker.read(dt_frame)
        claw.update(inp, max(dt_frame, 1e-6))

        while acc >= DT_FIXED:
            sim.space.step(DT_FIXED)
            acc -= DT_FIXED

        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, (28, 28, 32), (0, FLOOR_Y, SCREEN_W, SCREEN_H - FLOOR_Y))
        sim.space.debug_draw(draw_opts)
        draw_buckets(screen)
        draw_claw(screen, claw)
        draw_weight(screen, font, claw)
        draw_camera_panel(screen, tracker)

        fps_txt = font.render(f"fps {clock.get_fps():5.1f}", True, (200, 200, 210))
        screen.blit(fps_txt, (12, SCREEN_H - 32))

        pygame.display.flip()

    tracker.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
