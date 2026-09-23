# main.py
import sys
import math
import random
import numpy as np
import pygame

from neuron import Neuron
from creature import Creature

# ---------------- config ----------------
SEED = 1
N_INPUT = 3
N_OUTPUT = 3
N_HIDDEN = 10
N_SYNAPSES = 30
SCREEN_W, SCREEN_H = 1000, 640
PANEL_W = 260
WORLD_W = SCREEN_W - PANEL_W
MARGIN = 60                          # keep input/output columns off the screen edges
TICKS_PER_SECOND = 60
KICK_EVERY = 0                       # ticks between auto-kicks on a random input (0 = never)
FLASH_TICKS = 4
HISTORY = 200
VELOCITY = 70.0

random.seed(SEED)
rng = np.random.default_rng(SEED)

INPUT_COLOR = (90, 230, 140)         # green
OUTPUT_COLOR = (255, 140, 90)        # orange
HIDDEN_COLOR = (60, 110, 255)        # blue


def delay_from_distance(a, b):
    return max(1, round(math.dist(a.pos, b.pos) / VELOCITY))


# ---------------- build the brain ----------------
def column_positions(n, x, y_top, y_bottom):
    """Evenly space n neurons down a fixed vertical column at x."""
    if n == 1:
        return [(x, (y_top + y_bottom) // 2)]
    step = (y_bottom - y_top) / (n - 1)
    return [(x, round(y_top + i * step)) for i in range(n)]

def make_sane_neuron(pos, color, rng, is_input=False, is_output=False, max_resample=20, **kwargs):
    for _ in range(max_resample):
        n = Neuron(pos, color, rng=rng, is_input_neuron=is_input, is_output_neuron=is_output, **kwargs)
        r = n.sensitivity_report()
        if r["can_fire"] and not n.self_sustains():
            return n
    raise RuntimeError(f"couldn't draw a sane neuron after {max_resample} tries")

input_positions = column_positions(N_INPUT, MARGIN, MARGIN, SCREEN_H - MARGIN)
output_positions = column_positions(N_OUTPUT, WORLD_W - MARGIN, MARGIN, SCREEN_H - MARGIN)

input_neurons = [
    make_sane_neuron(pos=pos, color=INPUT_COLOR, rng=rng,
           threshold_margin=0.05, input_gain=8.0, is_input=True)
    for pos in input_positions
]
output_neurons = [
    make_sane_neuron(pos=pos, color=OUTPUT_COLOR, rng=rng,
           threshold_margin=0.05, input_gain=8.0, is_output=True)
    for pos in output_positions
]
hidden_neurons = [
    make_sane_neuron(
        pos=(random.randint(MARGIN + 60, WORLD_W - MARGIN - 60), random.randint(30, SCREEN_H - 30)),
        color=HIDDEN_COLOR, rng=rng,
        threshold_margin=0.05, input_gain=8.0,
    )
    for _ in range(N_HIDDEN)
]

neurons = input_neurons + hidden_neurons + output_neurons
N_NEURONS = len(neurons)

# wire hidden<->hidden and input->hidden freely; output neurons are capped
# at one incoming synapse by add_synapse itself, so this loop can call it blindly
wireable_sources = input_neurons + hidden_neurons
wireable_targets = hidden_neurons + output_neurons

for n in hidden_neurons:
    n.is_inhibitory = rng.random() < 0.2   # ~20% inhibitory, a common cortical ratio ballpark

made = 0
attempts = 0
while made < N_SYNAPSES and attempts < N_SYNAPSES * 30:
    a = random.choice(wireable_sources)
    b = random.choice(wireable_targets)
    before = len(a.outgoing_synapses)
    raw_weight = float(rng.uniform(0.4, 1.2))   # magnitude only, always positive draw
    weight = -raw_weight if a.is_inhibitory else raw_weight
    a.add_synapse(
        b,
        weight,
        delay=delay_from_distance(a, b),
    )
    if len(a.outgoing_synapses) > before:
        made += 1
    attempts += 1


creature = Creature(neurons)

# ---------------- bookkeeping ----------------
tick = 0
paused = False
fire_history = [[] for _ in neurons]
flash = [0] * N_NEURONS
event_log = []


def log(msg):
    event_log.append((tick, msg))
    if len(event_log) > 14:
        event_log.pop(0)


def brain_tick(kick=False):
    """One full creature step, plus a manual kick on an input neuron."""
    global tick

    for n in input_neurons:
        n.fired = False

    if kick and input_neurons:
        n = random.choice(input_neurons)
        i = neurons.index(n)
        n.fired = True
        log(f"kick -> input {i}")

    creature.brain_tick()

    for i, n in enumerate(neurons):
        fire_history[i].append(1 if n.fired else 0)
        if len(fire_history[i]) > HISTORY:
            fire_history[i].pop(0)
        if n.fired:
            flash[i] = FLASH_TICKS
            log(f"{'input' if n.is_input_neuron else 'output' if n.is_output_neuron else 'neuron'} {i} FIRED")
        elif flash[i] > 0:
            flash[i] -= 1
    tick += 1


def firing_rate(i):
    h = fire_history[i]
    return sum(h) / len(h) if h else 0.0


def health_report():
    dead = [i for i in range(N_NEURONS) if len(fire_history[i]) >= 50 and firing_rate(i) == 0.0]
    stuck = [i for i in range(N_NEURONS) if len(fire_history[i]) >= 50 and firing_rate(i) > 0.9]
    return dead, stuck


# ---------------- startup diagnostics (terminal) ----------------
print(f"{N_INPUT} input, {N_HIDDEN} hidden, {N_OUTPUT} output\n")
print("--- neuron sensitivity ---")
bad = 0
for i, n in enumerate(neurons):
    sigs = [n._fire_signal(p, 0.0) for p in (0.5, 1.0, 2.0)]
    ok = sigs[1] > n.threshold
    bad += not ok
    kind = "IN " if n.is_input_neuron else "OUT" if n.is_output_neuron else "hid"
    print(f"n{i:2d} [{kind}] thr={n.threshold:.2f} out@0.5/1/2 = "
          f"{sigs[0]:.2f}/{sigs[1]:.2f}/{sigs[2]:.2f} {'OK' if ok else 'BAD'}")
print(f"{bad} of {len(neurons)} neurons failed\n")

for n in output_neurons:
    i = neurons.index(n)
    print(f"output n{i}: incoming synapses = {len(n.incoming_synapses)} (should be 0 or 1)")
print(f"\nwired: {made}/{N_SYNAPSES} synapses after {attempts} attempts\n")

for i, n in enumerate(neurons):
    outs = [neurons.index(s.receiver) for s in n.outgoing_synapses]
    print(f"n{i} ({'INH' if getattr(n, 'is_inhibitory', False) else 'exc'}) -> {outs}")

# ---------------- pygame ----------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Brain test")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 14)
small = pygame.font.SysFont("consolas", 12)

tick_timer = 0.0
running = True

for n in neurons:
    print(n.self_sustains(1.0))
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
                brain_tick(kick=True)
            elif event.key == pygame.K_RIGHT and paused:
                brain_tick()
            elif event.key == pygame.K_UP:
                TICKS_PER_SECOND = min(60, TICKS_PER_SECOND + 2)
            elif event.key == pygame.K_DOWN:
                TICKS_PER_SECOND = max(1, TICKS_PER_SECOND - 2)

    if not paused:
        tick_timer += dt
        step = 1.0 / TICKS_PER_SECOND
        while tick_timer >= step:
            tick_timer -= step
            kick = KICK_EVERY > 0 and tick % KICK_EVERY == 0
            brain_tick(kick=kick)
            print("tick ",tick)
            for n in neurons:
                print(f"pot={n.potential:.2f} trace={n.spike_trace:.2f} thr={n.threshold:.2f} raw={n.last_fire_signal:.2f} fired={n.fired}")
        
    # ---------------- draw ----------------
    screen.fill((8, 8, 14))

    # faint column guides so input/output alignment reads clearly
    pygame.draw.line(screen, (30, 45, 35), (MARGIN, 0), (MARGIN, SCREEN_H), 1)
    pygame.draw.line(screen, (45, 35, 30), (WORLD_W - MARGIN, 0), (WORLD_W - MARGIN, SCREEN_H), 1)

    for s in creature.all_synapses:
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

        if s.spike:
            pygame.draw.line(screen, bright, (x1, y1), (x2, y2), 2)

    for i, n in enumerate(neurons):
        firing = flash[i] > 0
        is_io = n.is_input_neuron or n.is_output_neuron
        radius = (10 if is_io else 8) + (3 if firing else 0)
        base = (255, 255, 255) if firing else n.color

        pygame.draw.circle(screen, base, n.pos, radius)
        if is_io:                                        # outline so IO neurons stand out
            pygame.draw.circle(screen, (255, 255, 255), n.pos, radius, 2)

        pot = max(0.0, min(n.potential / max(n.threshold, 1e-6), 1.0))
        pygame.draw.circle(screen, (90, 255, 150), n.pos, radius + 4, 1 + int(pot * 3))

        label_txt = (f"I{i}" if n.is_input_neuron else f"O{i}" if n.is_output_neuron else str(i))
        label = small.render(label_txt, True, (230, 230, 230) if is_io else (200, 200, 210))
        label_x = n.pos[0] - 24 if n.is_input_neuron else n.pos[0] + 12
        screen.blit(label, (label_x, n.pos[1] - 7))

    # column headers
    text_h = font.render("INPUTS", True, INPUT_COLOR)
    screen.blit(text_h, (MARGIN - 30, 6))
    text_o = font.render("OUTPUTS", True, OUTPUT_COLOR)
    screen.blit(text_o, (WORLD_W - MARGIN - 30, 6))

    # ---------------- side panel ----------------
    pygame.draw.rect(screen, (16, 16, 26), (WORLD_W, 0, PANEL_W, SCREEN_H))
    y = 8

    def text(msg, color=(220, 220, 230), f=font):
        global y
        screen.blit(f.render(msg, True, color), (WORLD_W + 10, y))
        y += 18

    text(f"tick {tick}   {'PAUSED' if paused else 'running'}", (255, 255, 140))
    text(f"speed {TICKS_PER_SECOND} ticks/s")
    text("SPACE pause  K kick input", (140, 140, 160), small)
    text("RIGHT step  UP/DOWN speed", (140, 140, 160), small)
    y += 6

    dead, stuck = health_report()
    text(f"dead (0% fire): {len(dead)}", (255, 120, 120) if dead else (120, 220, 140))
    text(f"stuck (>90%):   {len(stuck)}", (255, 120, 120) if stuck else (120, 220, 140))
    y += 6

    text("firing rate per neuron", (255, 255, 140))
    for i, n in enumerate(neurons):
        r = firing_rate(i)
        bar = int(r * 100)
        tag = "I" if n.is_input_neuron else "O" if n.is_output_neuron else " "
        color = INPUT_COLOR if n.is_input_neuron else OUTPUT_COLOR if n.is_output_neuron else (90, 255, 150)
        pygame.draw.rect(screen, (40, 40, 60), (WORLD_W + 40, y + 2, 100, 9))
        pygame.draw.rect(screen, color, (WORLD_W + 40, y + 2, bar, 9))
        text(f"{tag}{i:2d}", (170, 170, 190), small)
        screen.blit(small.render(f"{r*100:4.0f}%", True, (200, 200, 210)), (WORLD_W + 148, y - 16))
    y += 4

    text("event log", (255, 255, 140))
    for t, msg in event_log[-8:]:
        text(f"{t:5d} {msg}", (170, 170, 190), small)

    pygame.display.update()


pygame.quit()

print(f"\nran {tick} ticks")
dead, stuck = health_report()
print("dead neurons :", dead or "none")
print("stuck neurons:", stuck or "none")
for i, n in enumerate(neurons):
    kind = "IN " if n.is_input_neuron else "OUT" if n.is_output_neuron else "hid"
    print(f"  n{i:2d} [{kind}] rate={firing_rate(i)*100:5.1f}%  last raw fire signal={n.last_fire_signal:.2f}")
sys.exit()