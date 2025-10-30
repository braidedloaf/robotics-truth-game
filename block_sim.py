import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import pygame
import pymunk
import pymunk.pygame_util


SCREEN_W = 960
SCREEN_H = 600
PPM = 1.0  # pixels per meter visual scale kept simple
BG_COLOR = (18, 18, 20)
FLOOR_Y = SCREEN_H - 40
GRAVITY_Y = 1400.0  # pixels per second^2
DENSITY = 0.0008  # mass per pixel^2 to keep numbers practical
SPAWN_BURST = 5
FPS = 120


@dataclass
class HandPoint:
    x: float  # screen pixels
    y: float  # screen pixels
    strength: float = 1.0  # 0 to 2 suggested

class HandTracker:
    """
    Placeholder for MediaPipe integration.
    Implement start, read, stop. read returns a list of HandPoint in screen pixels.
    """
    def __init__(self, use_camera: bool = False):
        self.use_camera = use_camera
        self._started = False

    def start(self) -> None:
        # Wire up cv2.VideoCapture and MediaPipe Hands here later
        self._started = True

    def read(self) -> List[HandPoint]:
        # Return no hands for now. Later, map normalized landmarks to pixel space.
        return []

    def stop(self) -> None:
        self._started = False



@dataclass
class BoxSpec:
    w: float
    h: float
    mass: float
    color: Tuple[int, int, int]

class PhysicsSim:
    def __init__(self):
        self.space = pymunk.Space(threaded=False)
        self.space.gravity = (0.0, GRAVITY_Y)
        self.draw_options = None  # set after pygame init
        self.blocks: List[pymunk.Shape] = []
        self._setup_bounds()

    def _setup_bounds(self):
        static_body = self.space.static_body
        segs = []
        pad = 10
        # Floor
        segs.append(pymunk.Segment(static_body, (pad, FLOOR_Y), (SCREEN_W - pad, FLOOR_Y), 4))
        # Left wall
        segs.append(pymunk.Segment(static_body, (pad, 0), (pad, FLOOR_Y), 4))
        # Right wall
        segs.append(pymunk.Segment(static_body, (SCREEN_W - pad, 0), (SCREEN_W - pad, FLOOR_Y), 4))
        for s in segs:
            s.elasticity = 0.12
            s.friction = 0.9
            s.color = (90, 90, 100, 255)
        self.space.add(*segs)

    def add_box(self, x: float, y: float, w: float, h: float, density: float = DENSITY) -> pymunk.Shape:
        area = w * h
        mass = max(1.0, density * area)
        moment = pymunk.moment_for_box(mass, (w, h))
        body = pymunk.Body(mass, moment)
        body.position = x, y
        shape = pymunk.Poly.create_box(body, (w, h))
        shape.friction = 0.7
        shape.elasticity = 0.08
        # Color by mass percentile heuristic
        c = self._mass_color(mass)
        shape.color = (*c, 255)
        self.space.add(body, shape)
        self.blocks.append(shape)
        return shape

    def _mass_color(self, mass: float) -> Tuple[int, int, int]:
        # Map mass roughly into blue to red
        t = max(0.0, min(1.0, (mass - 20.0) / 300.0))
        r = int(40 + 180 * t)
        g = int(120 - 80 * t)
        b = int(220 - 180 * t)
        return (r, g, b)

    def clear_dynamic(self):
        for s in list(self.blocks):
            if s.body.body_type == pymunk.Body.DYNAMIC:
                self.space.remove(s, s.body)
        self.blocks.clear()

    def step(self, dt: float, hand_points: List[HandPoint]):
        # Optional hand influence. Apply radial forces near each hand.
        for hp in hand_points:
            self._apply_hand_field(hp)
        self.space.step(dt)

    def _apply_hand_field(self, hp: HandPoint):
        # Simple repulsive field. Tune later when MediaPipe lands.
        radius = 140.0
        k = 22000.0 * hp.strength
        for s in self.blocks:
            b = s.body
            dx = b.position.x - hp.x
            dy = b.position.y - hp.y
            r2 = dx * dx + dy * dy
            if r2 < 1.0:
                continue
            r = math.sqrt(r2)
            if r > radius:
                continue
            nx = dx / r
            ny = dy / r
            # strength falls off with distance
            mag = k * (1.0 - r / radius)
            fx = nx * mag
            fy = ny * mag
            b.apply_force_at_world_point((fx, fy), b.position)


def spawn_burst(sim: PhysicsSim, n: int = SPAWN_BURST):
    for _ in range(n):
        w = random.uniform(28, 72)
        h = random.uniform(28, 72)
        x = random.uniform(60, SCREEN_W - 60)
        y = random.uniform(-80, -10)
        sim.add_box(x, y, w, h)


def spawn_heavy(sim: PhysicsSim):
    # Heavier via larger size and density factor
    w = random.uniform(80, 120)
    h = random.uniform(50, 100)
    x = random.uniform(100, SCREEN_W - 100)
    y = random.uniform(-140, -40)
    sim.add_box(x, y, w, h, density=DENSITY * 2.5)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Block Gravity Sim with Hand Input Seam")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)

    sim = PhysicsSim()
    draw_opts = pymunk.pygame_util.DrawOptions(screen)
    sim.draw_options = draw_opts

    hand = HandTracker(use_camera=False)
    hand.start()

    # Initial spawn
    spawn_burst(sim, 12)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # Input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    spawn_burst(sim, SPAWN_BURST)
                elif event.key == pygame.K_b:
                    spawn_heavy(sim)
                elif event.key == pygame.K_c:
                    sim.clear_dynamic()

        hand_points = hand.read()

        # Physics
        sim.step(dt, hand_points)

        # Draw
        screen.fill(BG_COLOR)
        # ground stripe
        pygame.draw.rect(screen, (28, 28, 32), pygame.Rect(0, FLOOR_Y, SCREEN_W, SCREEN_H - FLOOR_Y))
        sim.space.debug_draw(sim.draw_options)

        # Optional visualize hand fields
        for hp in hand_points:
            pygame.draw.circle(screen, (230, 230, 255), (int(hp.x), int(hp.y)), 6, 0)
            pygame.draw.circle(screen, (90, 90, 120), (int(hp.x), int(hp.y)), 140, 1)

        # HUD
        fps_txt = font.render(f"fps {clock.get_fps():5.1f}", True, (200, 200, 210))
        cnt_txt = font.render(f"blocks {len(sim.blocks):3d}", True, (200, 200, 210))
        help_txt = font.render("space add 5   b add heavy   c clear   esc quit", True, (150, 150, 160))
        screen.blit(fps_txt, (10, 8))
        screen.blit(cnt_txt, (10, 26))
        screen.blit(help_txt, (10, 46))

        pygame.display.flip()

    hand.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
