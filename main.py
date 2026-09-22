# test_brain.py
import sys
import math
import random
import numpy as np
import pygame
from neuron import Neuron

# ---------------- config ----------------
SEED = 42
N_NEURONS = 20
N_SYNAPSES = 60
SCREEN_W, SCREEN_H = 1000, 640
PANEL_W = 260                       # right-hand log panel
WORLD_W = SCREEN_W - PANEL_W
TICKS_PER_SECOND = 6                # slow enough to watch spikes travel
KICK_EVERY = 0                     # ticks between random external kicks (0 = never)
FLASH_TICKS = 4                     # how long a firing neuron stays bright
HISTORY = 200                       # ticks kept for firing-rate stats
VELOCITY = 70.0                     # pixels per tick; lower = slower conduction, bigger delays

random.seed(SEED)
rng = np.random.default_rng(SEED)


def delay_from_distance(a, b):
    return max(1, round(math.dist(a.pos, b.pos) / VELOCITY))


# ---------------- build the brain ----------------
neurons = [
    Neuron(
        pos=(random.randint(30, WORLD_W - 30), random.randint(30, SCREEN_H - 30)),
        color=(60, 110, 255),
        rng=rng,
        threshold_margin=0.15,
        input_gain=3.0,
    )
    for _ in range(N_NEURONS)
]

made = 0
while made < N_SYNAPSES:
    a, b = random.sample(neurons, 2)
    before = len(a.outgoing_synapses)
    a.add_synapse(
        b,
        float(rng.uniform(-0.4, 1.2)),        # mostly excitatory, some inhibitory
        delay=delay_from_distance(a, b),      # spatial delay
    )
    if len(a.outgoing_synapses) > before:
        made += 1

all_synapses = [s for n in neurons for s in n.outgoing_synapses]

# ---------------- bookkeeping ----------------
tick = 0
paused = False
fire_history = [[] for _ in neurons]      # per-neuron list of 0/1 over recent ticks
flash = [0] * N_NEURONS                   # ticks remaining of bright display
event_log = []                            # (tick, message)


def log(msg):
    event_log.append((tick, msg))
    if len(event_log) > 14:
        event_log.pop(0)


def brain_tick(kick=False):
    """One full network step: advance the wires, read arrivals, send new spikes."""
    global tick
        
    if kick:
        i = random.randrange(N_NEURONS)
        for s in neurons[i].incoming_synapses:
            s.spike = True
        log(f"kick -> neuron {i}")

    for n in neurons:
        n.process()
    for n in neurons:
        n.forward()
    for s in all_synapses:
        s.advance()

    for i, n in enumerate(neurons):
        fire_history[i].append(1 if n.fired else 0)
        if len(fire_history[i]) > HISTORY:
            fire_history[i].pop(0)
        if n.fired:
            flash[i] = FLASH_TICKS
            log(f"neuron {i} FIRED")
        elif flash[i] > 0:
            flash[i] -= 1
    tick += 1



def firing_rate(i):
    h = fire_history[i]
    return sum(h) / len(h) if h else 0.0


def health_report():
    """Flag neurons that are dead (never fire) or saturated (always fire)."""
    dead = [i for i in range(N_NEURONS) if len(fire_history[i]) >= 50 and firing_rate(i) == 0.0]
    stuck = [i for i in range(N_NEURONS) if len(fire_history[i]) >= 50 and firing_rate(i) > 0.9]
    return dead, stuck


# ---------------- startup diagnostics (terminal) ----------------
print("--- neuron sensitivity (can potential ~1.0, about two spikes, fire it?) ---")
bad = 0
for i, n in enumerate(neurons):
    sigs = [n._fire_signal(p, 0.0) for p in (0.5, 1.0, 2.0)]
    ok = sigs[1] > n.threshold
    bad += not ok
    print(f"n{i:2d} thr={n.threshold:.2f} out@0.5/1/2 = "
          f"{sigs[0]:.2f}/{sigs[1]:.2f}/{sigs[2]:.2f} {'OK' if ok else 'BAD'}")
print(f"{bad} of {len(neurons)} neurons failed\n")

print("--- wiring (in=0 means that neuron can never fire) ---")
for i, n in enumerate(neurons):
    print(f"n{i:2d} in={len(n.incoming_synapses):2d} out={len(n.outgoing_synapses):2d}")
delays = [s.delay for s in all_synapses]
print(f"\ndelays: min={min(delays)} max={max(delays)} mean={sum(delays)/len(delays):.1f}\n")

# ---------------- pygame ----------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Brain test")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 14)
small = pygame.font.SysFont("consolas", 12)

tick_timer = 0.0
running = True

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_k:
                brain_tick(kick=True)           # manual kick + step
            elif event.key == pygame.K_RIGHT and paused:
                brain_tick()                    # single-step while paused
            elif event.key == pygame.K_UP:
                TICKS_PER_SECOND = min(60, TICKS_PER_SECOND + 2)
            elif event.key == pygame.K_DOWN:
                TICKS_PER_SECOND = max(1, TICKS_PER_SECOND - 2)

    # advance the simulation at its own rate, separate from the 60 fps drawing
    if not paused:
        tick_timer += dt
        step = 1.0 / TICKS_PER_SECOND
        while tick_timer >= step:
            tick_timer -= step
            kick = False
            brain_tick(kick=kick)

    # ---------------- draw ----------------
    screen.fill((8, 8, 14))

    # synapses: dim line always, bright dot for each spike travelling along it
    for s in all_synapses:
        x1, y1 = s.sender.pos
        x2, y2 = s.receiver.pos
        excit = s.weight >= 0
        dim = (60, 60, 75) if excit else (75, 50, 50)
        bright = (255, 220, 90) if excit else (255, 90, 90)

        pygame.draw.line(screen, dim, (x1, y1), (x2, y2), 1)

        for frac in s.in_flight():
            px = x1 + (x2 - x1) * frac
            py = y1 + (y2 - y1) * frac
            radius = 3 + int(min(abs(s.weight), 1.5) * 2)
            pygame.draw.circle(screen, bright, (int(px), int(py)), radius)

        if s.spike:                              # spike landed this tick
            pygame.draw.line(screen, bright, (x1, y1), (x2, y2), 2)

    # neurons: glow when they fire, ring shows the current potential
    for i, n in enumerate(neurons):
        firing = flash[i] > 0
        radius = 8 + (3 if firing else 0)
        base = (255, 255, 255) if firing else n.color
        pygame.draw.circle(screen, base, n.pos, radius)
        pot = max(0.0, min(n.potential / max(n.threshold, 1e-6), 1.0))
        pygame.draw.circle(screen, (90, 255, 150), n.pos, radius + 4, 1 + int(pot * 3))
        label = small.render(str(i), True, (200, 200, 210))
        screen.blit(label, (n.pos[0] + 10, n.pos[1] - 16))

    # ---------------- side panel ----------------
    pygame.draw.rect(screen, (16, 16, 26), (WORLD_W, 0, PANEL_W, SCREEN_H))
    y = 8

    def text(msg, color=(220, 220, 230), f=font):
        global y
        screen.blit(f.render(msg, True, color), (WORLD_W + 10, y))
        y += 18

    text(f"tick {tick}   {'PAUSED' if paused else 'running'}", (255, 255, 140))
    text(f"speed {TICKS_PER_SECOND} ticks/s")
    text("SPACE pause  K kick", (140, 140, 160), small)
    text("RIGHT step  UP/DOWN speed", (140, 140, 160), small)
    y += 6

    dead, stuck = health_report()
    text(f"dead (0% fire): {len(dead)}", (255, 120, 120) if dead else (120, 220, 140))
    text(f"stuck (>90%):   {len(stuck)}", (255, 120, 120) if stuck else (120, 220, 140))
    y += 6

    text("firing rate per neuron", (255, 255, 140))
    for i in range(N_NEURONS):
        r = firing_rate(i)
        bar = int(r * 100)
        pygame.draw.rect(screen, (40, 40, 60), (WORLD_W + 40, y + 2, 100, 9))
        pygame.draw.rect(screen, (90, 255, 150), (WORLD_W + 40, y + 2, bar, 9))
        text(f"{i:2d}", (170, 170, 190), small)
        screen.blit(small.render(f"{r*100:4.0f}%", True, (200, 200, 210)), (WORLD_W + 148, y - 16))
    y += 4

    text("event log", (255, 255, 140))
    for t, msg in event_log[-8:]:
        text(f"{t:5d} {msg}", (170, 170, 190), small)

    pygame.display.update()

pygame.quit()

# ---------------- final report in the terminal ----------------
print(f"\nran {tick} ticks")
dead, stuck = health_report()
print("dead neurons :", dead or "none")
print("stuck neurons:", stuck or "none")
for i, n in enumerate(neurons):
    print(f"  n{i:2d} rate={firing_rate(i)*100:5.1f}%  last raw fire signal={n.last_fire_signal:.2f}")
sys.exit()