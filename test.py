# test.py

import sys
import math
import random
import numpy as np
import pygame

from neuron import Neuron
from creature import Creature

# ---------------- window / world ----------------
SCREEN_W, SCREEN_H = 1200, 720
PANEL_W = 300
WORLD_W = SCREEN_W - PANEL_W
FPS = 60

INPUT_COLOR = (90, 230, 140)
OUTPUT_COLOR = (255, 140, 90)
HIDDEN_COLOR = (60, 110, 255)
INHIBITORY_RING = (255, 90, 90)

# ---------------- tunable constants (edit live via the panel, see CONTROLS) ----------------
controls = {
    "velocity": 70.0,        # px/tick -> synapse delay = max(1, dist/velocity)
    "chem_decay": 0.95,      # per-tick multiplier on chem_inputs (0..1, closer to 1 = lingers longer)
    "chem_range": 120.0,     # px; falloff = exp(-dist / chem_range)
    "chem_clip": 5.0,        # max value any single chem_inputs channel can reach
    "default_weight": 0.8,   # weight used for synapses drawn in the editor
    "chem_channel_view": 0,  # which of the neuron's chemical channels the heatmap shows
}
CONTROL_ORDER = ["velocity", "chem_decay", "chem_range", "chem_clip", "default_weight", "chem_channel_view"]
CONTROL_STEP = {
    "velocity": 5.0, "chem_decay": 0.01, "chem_range": 10.0,
    "chem_clip": 0.5, "default_weight": 0.1, "chem_channel_view": 1,
}


def delay_from_distance(a, b):
    return max(1, round(math.dist(a.pos, b.pos) / controls["velocity"]))


def chem_tick(creature):
    """Diffuse chem_release -> chem_inputs across the whole creature, one tick's worth.
    Self-excluded (a neuron never senses its own release), exponential falloff (bounded,
    no 1/distance blow-up), clipped once per tick after the full sum -- see chat history
    for why each of those three matters."""
    decay = controls["chem_decay"]
    rng_ = controls["chem_range"]
    clip = controls["chem_clip"]
    for n in creature.neurons:
        n.chem_inputs *= decay
        for n2 in creature.neurons:
            if n2 is n:
                continue
            if not np.any(n2.chem_release):
                continue
            dist = math.hypot(n.pos[0] - n2.pos[0], n.pos[1] - n2.pos[1])
            n.chem_inputs += n2.chem_release * math.exp(-dist / rng_)
        np.clip(n.chem_inputs, 0.0, clip, out=n.chem_inputs)


def make_sane_neuron(pos, color, rng, is_input=False, is_output=False, max_resample=20, **kwargs):
    for _ in range(max_resample):
        n = Neuron(pos, color, rng=rng, is_input_neuron=is_input, is_output_neuron=is_output, **kwargs)
        r = n.sensitivity_report()
        if r["can_fire"] and not n.self_sustains():
            return n
    raise RuntimeError(f"couldn't draw a sane neuron after {max_resample} tries")
SEED = 10
random.seed(SEED)
rng = np.random.default_rng(SEED)

neuron_palette = [make_sane_neuron(pos=(0,0),
                                   color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)),
                                   rng=rng,
                                   threshold_margin=0.15,
                                   homeostasis_tau=float(rng.uniform(0.5, 0.9)),
                                   potential_leak=float(rng.uniform(0.1,0.3)),
                                   spike_leak=float(rng.uniform(0.1,0.3)),
                                   input_gain=float(rng.uniform(0.5,3)) ) for _ in range(15)]

import copy
def make_neuron(pos, kind, rng):
    """kind: 'input' | 'output' | 'hidden'. Resamples until sane (can_fire, not self_sustains),
    same guard main.py uses, so the playground can't hand you a permanently dead/pacemaking neuron."""
    is_input = kind == "input"
    is_output = kind == "output"
    n = copy.deepcopy(random.choice(neuron_palette))
    color = INPUT_COLOR if is_input else OUTPUT_COLOR if is_output else n.color
    n.color = color
    n.pos = pos
    n.rng = rng
    if is_input:
        n.is_input_neuron = True
    if is_output:
        n.is_output_neuron = True
    #for _ in range(20):

    #n = Neuron(pos, color, rng=rng, is_input_neuron=is_input, is_output_neuron=is_output,
    #            homeostasis_tau=0.9,
    #            spike_leak=0.2,
    #            potential_leak=0.2,
    #            threshold_margin=0.15, input_gain=0.5)
    r = n.sensitivity_report()
    if r["can_fire"] and not n.self_sustains():
        n.is_inhibitory = False
        return n
    raise RuntimeError("couldn't draw a sane neuron after 20 tries")


def reset_neuron_state(n):
    n.potential = 0.0
    n.spike_trace = 0.0
    n.fired = False
    n.last_fire_signal = 0.0
    n.rate_estimate = n.target_rate
    n.threshold = float(np.clip(n.target_rest + n.threshold_margin, n.min_threshold, n.max_threshold))
    n.chem_inputs[:] = 0.0
    n.chem_release[:] = 0.0
    n.incoming = []
    for s in n.incoming_synapses:
        s.spike = False
        s.buffer[:] = False


def reset_all():
    for n in creature.neurons:
        reset_neuron_state(n)


# ---------------- state ----------------
rng = np.random.default_rng(0)
#creature = Creature([])
creature = Creature.load("saves/gen_0/best_of_creature_0.json")
tick = 0
paused = True
selected = None            # neuron currently being dragged from, for wiring
place_kind = "hidden"      # what clicking empty space creates: input/hidden/output
show_chem_heatmap = True
show_synapse_labels = False
selected_control = 0
flash = {}                 # id(neuron) -> ticks remaining of "just fired" glow
event_log = []


def log(msg):
    event_log.append((tick, msg))
    if len(event_log) > 12:
        event_log.pop(0)


def neuron_at(pos, radius=14):
    for n in creature.neurons:
        if math.hypot(n.pos[0] - pos[0], n.pos[1] - pos[1]) <= radius:
            return n
    return None


def add_neuron_at(pos):
    n = make_neuron(pos, place_kind, rng)
    creature.neurons.append(n)
    creature.input_neurons = [x for x in creature.neurons if x.is_input_neuron]
    creature.output_neurons = [x for x in creature.neurons if x.is_output_neuron]
    log(f"placed {place_kind} neuron ({len(creature.neurons) - 1})")
    return n


def remove_neuron(n):
    """Fully detach and drop a neuron: clear its own synapses and anyone else's synapses into it."""
    for s in list(n.outgoing_synapses):
        n.delete_synapse(s.receiver)
    for other in creature.neurons:
        if other is n:
            continue
        for s in list(other.outgoing_synapses):
            if s.receiver is n:
                other.delete_synapse(n)
    creature.neurons.remove(n)
    creature.input_neurons = [x for x in creature.neurons if x.is_input_neuron]
    creature.output_neurons = [x for x in creature.neurons if x.is_output_neuron]
    log("removed neuron")


def toggle_inhibitory(n):
    if n.is_input_neuron or n.is_output_neuron:
        log("input/output neurons can't be inhibitory")
        return
    n.is_inhibitory = not n.is_inhibitory
    # re-sign existing outgoing synapses to match, same convention main.py uses
    for s in n.outgoing_synapses:
        mag = abs(s.weight)
        s.weight = -mag if n.is_inhibitory else mag
    log(f"neuron is now {'inhibitory' if n.is_inhibitory else 'excitatory'}")


def wire(a, b):
    if a is b:
        return
    weight = controls["default_weight"]
    if a.is_inhibitory:
        weight = -abs(weight)
    a.add_synapse(b, weight, delay=delay_from_distance(a, b))
    log(f"wired -> {'ok' if any(s.receiver is b for s in a.outgoing_synapses) else 'rejected (rule/cap)'}")


def unwire(a, b):
    a.delete_synapse(b)
    log("unwired")


def step():
    global tick
    creature.brain_tick()
    for n in creature.neurons:
        if n.fired:
            flash[id(n)] = 6
        elif flash.get(id(n), 0) > 0:
            flash[id(n)] -= 1
    tick += 1


def adjust_control(delta):
    key = CONTROL_ORDER[selected_control]
    controls[key] = max(0.0, controls[key] + delta * CONTROL_STEP[key])
    if key == "chem_channel_view":
        max_ch = max((n.number_of_input_chemicals for n in creature.neurons), default=1)
        controls[key] = int(np.clip(controls[key], 0, max_ch - 1))


# ---------------- pygame ----------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("neuron playground")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 14)
small = pygame.font.SysFont("consolas", 12)
tiny = pygame.font.SysFont("consolas", 11)

heatmap_surf = pygame.Surface((WORLD_W, SCREEN_H), pygame.SRCALPHA)


def draw_heatmap():
    """Coarse grid heatmap of one chemical channel's concentration, sampled by nearest neuron."""
    heatmap_surf.fill((0, 0, 0, 0))
    if not creature.neurons or not show_chem_heatmap:
        screen.blit(heatmap_surf, (0, 0))
        return
    ch = controls["chem_channel_view"]
    cell = 20
    for gx in range(0, WORLD_W, cell):
        for gy in range(0, SCREEN_H, cell):
            cx, cy = gx + cell / 2, gy + cell / 2
            total = 0.0
            for n in creature.neurons:
                if not np.any(n.chem_release):
                    continue
                dist = math.hypot(n.pos[0] - cx, n.pos[1] - cy)
                total += n.chem_release[ch] * math.exp(-(dist) / controls["chem_range"]) 
            if total > 0.01:
                alpha = int(min(total, 1.0) * 160)
                pygame.draw.rect(heatmap_surf, (255, 140, 0, alpha), (gx, gy, cell, cell))
    screen.blit(heatmap_surf, (0, 0))


def draw():
    screen.fill((8, 8, 14))
    draw_heatmap()

    for n in creature.neurons:
        for s in n.outgoing_synapses:
            x1, y1 = s.sender.pos
            x2, y2 = s.receiver.pos
            excit = s.weight >= 0
            dim = (60, 60, 75) if excit else (75, 50, 50)
            bright = (255, 220, 90) if excit else (255, 90, 90)
            pygame.draw.line(screen, dim, (x1, y1), (x2, y2), 1)
            for frac in s.in_flight():
                px, py = x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac
                pygame.draw.circle(screen, bright, (int(px), int(py)), 4)
            if s.spike:
                pygame.draw.line(screen, bright, (x1, y1), (x2, y2), 2)
            if show_synapse_labels:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                lbl = tiny.render(f"{s.weight:+.2f} d{s.delay}", True, (180, 180, 190))
                screen.blit(lbl, (mx, my))

    for n in creature.neurons:
        firing = flash.get(id(n), 0) > 0
        is_io = n.is_input_neuron or n.is_output_neuron
        radius = (10 if is_io else 8) + (3 if firing else 0)
        base = (255, 255, 255) if firing else n.color
        pygame.draw.circle(screen, base, n.pos, radius)
        if is_io:
            pygame.draw.circle(screen, (255, 255, 255), n.pos, radius, 2)
        if n.is_inhibitory:
            pygame.draw.circle(screen, INHIBITORY_RING, n.pos, radius + 3, 1)
        if n is selected:
            pygame.draw.circle(screen, (255, 255, 0), n.pos, radius + 6, 2)
        pot = max(0.0, min(n.potential / max(n.threshold, 1e-6), 1.0))
        pygame.draw.circle(screen, (90, 255, 150), n.pos, radius + 4, 1 + int(pot * 3))
        idx = creature.neurons.index(n)
        label_txt = f"I{idx}" if n.is_input_neuron else f"O{idx}" if n.is_output_neuron else str(idx)
        label = small.render(label_txt, True, (230, 230, 230) if is_io else (200, 200, 210))
        screen.blit(label, (n.pos[0] + 12, n.pos[1] - 7))

    if selected is not None:
        mx, my = pygame.mouse.get_pos()
        if mx < WORLD_W:
            pygame.draw.line(screen, (255, 255, 0), selected.pos, (mx, my), 1)

    draw_panel()
    pygame.display.update()


def draw_panel():
    pygame.draw.rect(screen, (16, 16, 26), (WORLD_W, 0, PANEL_W, SCREEN_H))
    y = [8]

    def text(msg, color=(220, 220, 230), f=font):
        screen.blit(f.render(msg, True, color), (WORLD_W + 10, y[0]))
        y[0] += 18

    text("NEURON PLAYGROUND", (255, 255, 140))
    text(f"tick {tick}  {'PAUSED' if paused else 'running'}")
    text(f"neurons: {len(creature.neurons)}   synapses: {len(creature.all_synapses)}")
    y[0] += 6

    text(f"place mode: {place_kind.upper()}", (140, 220, 220))
    text("1/2/3 = input/hidden/output", (150, 150, 165), small)
    text("click empty = place neuron", (150, 150, 165), small)
    text("click neuron, drag, release on", (150, 150, 165), small)
    text("  another = wire them", (150, 150, 165), small)
    text("X = delete hovered neuron", (150, 150, 165), small)
    text("U = unwire hovered pair (click+drag", (150, 150, 165), small)
    text("  then U while held) ", (150, 150, 165), small)
    text("I = toggle inhibitory on hovered", (150, 150, 165), small)
    text("K = fire hovered input neuron", (150, 150, 165), small)
    text("SPACE = pause  RIGHT = step", (150, 150, 165), small)
    text("R = reset all neuron state", (150, 150, 165), small)
    text("C = clear whole graph", (150, 150, 165), small)
    text("H = toggle chem heatmap", (150, 150, 165), small)
    text("L = toggle synapse labels", (150, 150, 165), small)
    y[0] += 6

    text("controls  (TAB select, -/+ adjust)", (255, 255, 140))
    for i, key in enumerate(CONTROL_ORDER):
        marker = ">" if i == selected_control else " "
        val = controls[key]
        vtxt = f"{val:.2f}" if isinstance(val, float) else str(val)
        text(f"{marker} {key}: {vtxt}", (200, 220, 200) if i == selected_control else (170, 170, 190), small)
    y[0] += 6

    if creature.neurons:
        text("hovered neuron", (255, 255, 140))
        mx, my = pygame.mouse.get_pos()
        hn = neuron_at((mx, my)) if mx < WORLD_W else None
        if hn is not None:
            idx = creature.neurons.index(hn)
            kind = "input" if hn.is_input_neuron else "output" if hn.is_output_neuron else "hidden"
            text(f"n{idx} [{kind}]{' INH' if hn.is_inhibitory else ''}", (200, 200, 210), small)
            text(f"thr={hn.threshold:.2f} pot={hn.potential:.2f}", (200, 200, 210), small)
            text(f"trace={hn.spike_trace:.2f} fired={hn.fired}", (200, 200, 210), small)
            chem_str = " ".join(f"{v:.2f}" for v in hn.chem_inputs)
            text(f"chem_in: [{chem_str}]", (200, 200, 210), small)
            rel_str = " ".join(f"{v:.2f}" for v in hn.chem_release)
            text(f"chem_out: [{rel_str}]", (200, 200, 210), small)
        else:
            text("(hover a neuron)", (140, 140, 160), small)
    y[0] += 6

    text("event log", (255, 255, 140))
    for t, msg in event_log[-6:]:
        text(f"{t:5d} {msg}", (170, 170, 190), tiny)


def handle_key(key):
    global paused, place_kind, selected_control, show_chem_heatmap, show_synapse_labels

    if key == pygame.K_SPACE:
        paused = not paused
    elif key == pygame.K_RIGHT and paused:
        step()
    elif key == pygame.K_1:
        place_kind = "input"
    elif key == pygame.K_2:
        place_kind = "hidden"
    elif key == pygame.K_3:
        place_kind = "output"
    elif key == pygame.K_r:
        reset_all()
        log("reset all neuron state")
    elif key == pygame.K_c:
        creature.neurons.clear()
        creature.input_neurons = []
        creature.output_neurons = []
        log("cleared graph")
    elif key == pygame.K_h:
        show_chem_heatmap = not show_chem_heatmap
    elif key == pygame.K_l:
        show_synapse_labels = not show_synapse_labels
    elif key == pygame.K_TAB:
        selected_control = (selected_control + 1) % len(CONTROL_ORDER)
    elif key in (pygame.K_MINUS, pygame.K_UNDERSCORE):
        adjust_control(-1)
    elif key in (pygame.K_EQUALS, pygame.K_PLUS):
        adjust_control(+1)
    elif key == pygame.K_x:
        mx, my = pygame.mouse.get_pos()
        hn = neuron_at((mx, my)) if mx < WORLD_W else None
        if hn is not None:
            remove_neuron(hn)
    elif key == pygame.K_i:
        mx, my = pygame.mouse.get_pos()
        hn = neuron_at((mx, my)) if mx < WORLD_W else None
        if hn is not None:
            toggle_inhibitory(hn)
    elif key == pygame.K_k:
        mx, my = pygame.mouse.get_pos()
        hn = neuron_at((mx, my)) if mx < WORLD_W else None
        if hn is not None and hn.is_input_neuron:
            if hn.fired == False:
                hn.fired = True
                log(f"kicked input n{creature.neurons.index(hn)}")
            else:
                hn.fired = False
                log(f"unkicked input n{creature.neurons.index(hn)}")
            if paused:
                step()   # force one tick so the kick actually propagates while paused
            
    elif key == pygame.K_u and selected is not None:
        mx, my = pygame.mouse.get_pos()
        hn = neuron_at((mx, my)) if mx < WORLD_W else None
        if hn is not None:
            unwire(selected, hn)


running = True
tick_timer = 0.0
TICKS_PER_SECOND = 3

while running:
    dt = clock.tick(FPS) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            else:
                handle_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.pos[0] < WORLD_W:
            hit = neuron_at(event.pos)
            if hit is not None:
                selected = hit
            else:
                add_neuron_at(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and selected is not None:
            if event.pos[0] < WORLD_W:
                hit = neuron_at(event.pos)
                if hit is not None and hit is not selected:
                    wire(selected, hit)
            selected = None

    if not paused:
        tick_timer += dt
        step_len = 1.0 / TICKS_PER_SECOND
        while tick_timer >= step_len:
            tick_timer -= step_len
            step()

    draw()

pygame.quit()
sys.exit()