# main.py
import sys
import math
import random
import numpy as np
import pygame

from neuron import Neuron
from creature import Creature
from finetune import Finetune

# ---------------- config ----------------
SEED = 1
N_INPUT = 3
N_OUTPUT = 3
N_HIDDEN = 10
N_SYNAPSES = 5
SCREEN_W, SCREEN_H = 1000, 640
PANEL_W = 260
WORLD_W = SCREEN_W - PANEL_W
MARGIN = 60
TICKS_PER_SECOND = 60
KICK_EVERY = 0
FLASH_TICKS = 4
HISTORY = 200
VELOCITY = 70.0

# --- NEAT / evaluation config ---
GENERATIONS = 100
BATCH_SIZE = 20
TRIAL_TICKS = 500          # ticks per evaluation trial; keep tight so wasteful routing costs fitness
CORRECT_OUTPUT_INDEX = 0  # which output index counts as "correct" for a single-pair test (unused by evaluate_multi)
FIRE_INPUT_INDEX = 1      # which input neuron to fire for a single-pair test (unused by evaluate_multi)

# input i must trigger output i, and only output i, for each pair below.
# (fire_input_index, correct_output_index)
MODULARITY_PAIRS = ((0, 0), (1, 1), (2, 2))

random.seed(SEED)
rng = np.random.default_rng(SEED)

INPUT_COLOR = (90, 230, 140)   # green
OUTPUT_COLOR = (255, 140, 90)  # orange
HIDDEN_COLOR = (60, 110, 255)  # blue


def delay_from_distance(a, b):
    return max(1, round(math.dist(a.pos, b.pos) / VELOCITY))


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


def make_sane_creature(rng, n_input, n_output, n_hidden, margin, world_w, screen_h,
                        input_color, output_color, hidden_color, n_synapses):
    input_positions = column_positions(n_input, margin, margin, screen_h - margin)
    output_positions = column_positions(n_output, world_w - margin, margin, screen_h - margin)

    input_neurons = [
        make_sane_neuron(pos=pos, color=input_color, rng=rng,
                          threshold_margin=0.05, input_gain=8.0, is_input=True)
        for pos in input_positions
    ]
    output_neurons = [
        make_sane_neuron(pos=pos, color=output_color, rng=rng,
                          threshold_margin=0.05, input_gain=8.0, is_output=True)
        for pos in output_positions
    ]
    hidden_neurons = [
        make_sane_neuron(
            pos=(random.randint(margin + 60, world_w - margin - 60), random.randint(30, screen_h - 30)),
            color=hidden_color, rng=rng,
            threshold_margin=0.05, input_gain=8.0,
        )
        for _ in range(n_hidden)
    ]

    neurons = input_neurons + hidden_neurons + output_neurons

    # wire hidden<->hidden and input->hidden freely; output neurons are capped
    # at one incoming synapse by add_synapse itself, so this loop can call it blindly
    wireable_sources = input_neurons + hidden_neurons
    wireable_targets = hidden_neurons + output_neurons

    for n in hidden_neurons:
        n.is_inhibitory = rng.random() < 0.2  # ~20% inhibitory, a common cortical ratio ballpark
    for n in input_neurons + output_neurons:
        n.is_inhibitory = False  # excitatory by convention; add_synapse only checks sender.is_inhibitory

    made = 0
    attempts = 0
    while made < n_synapses and attempts < n_synapses * 30:
        a = random.choice(wireable_sources)
        b = random.choice(wireable_targets)
        before = len(a.outgoing_synapses)
        raw_weight = float(rng.uniform(0.4, 1.2))
        weight = -raw_weight if a.is_inhibitory else raw_weight

        if not ((a.is_input_neuron and b.is_output_neuron) or (a.is_output_neuron and b.is_input_neuron) or (a.is_input_neuron and b.is_input_neuron)  or (a.is_output_neuron and b.is_output_neuron)):
            a.add_synapse(b, weight, delay=delay_from_distance(a, b))

        if len(a.outgoing_synapses) > before:
            made += 1
        attempts += 1

    creature = Creature(neurons)
    return creature, neurons, input_neurons, output_neurons, made, attempts


def log(msg):
    event_log.append((tick, msg))
    if len(event_log) > 14:
        event_log.pop(0)


def reset_neuron_state(n):
    """Wipe everything that would leak between trials, back to birth calibration."""
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


def reset_creature(creature):
    for n in creature.neurons:
        reset_neuron_state(n)


def evaluate(creature, trial_ticks=TRIAL_TICKS, correct_output_index=CORRECT_OUTPUT_INDEX,
             fire_input_index=FIRE_INPUT_INDEX):
    """Run one clean trial: fire an input once, score based on which output fires.
    +1 if the correct output fires, -1 if a wrong output fires first, 0 if neither fires."""
    reset_creature(creature)
    input_neurons = [n for n in creature.neurons if n.is_input_neuron]
    output_neurons = [n for n in creature.neurons if n.is_output_neuron]

    input_neurons[fire_input_index].fired = True

    score = 0.0
    for t in range(trial_ticks):
        creature.brain_tick()
        if t == 0:
            for n in input_neurons:
                n.fired = False  # single-tick pulse; don't re-fire on later passes

        if output_neurons[correct_output_index].fired:
            score = 1.0

        wrong_fired = any(
            n.fired for i, n in enumerate(output_neurons) if i != correct_output_index
        )
        if wrong_fired:
            score = -1.0
            break
    return score


def evaluate_multi(creature, trial_ticks=TRIAL_TICKS, pairs=MODULARITY_PAIRS):
    """Test each (fire_input_index, correct_output_index) pair independently and sum the scores.
    Each pair gets its own clean trial (evaluate() resets the creature at the start of each call),
    so results from one pair never leak into another within the same generation's scoring.
    A shared pathway that helps pair A but fires the wrong output on pair B nets out worse than
    two properly separated pathways, which is the pressure that should push toward modularity."""
    return sum(evaluate(creature, trial_ticks, correct_index, fire_index)
               for fire_index, correct_index in pairs)


def run_generation(seed_creature, batch_size=BATCH_SIZE, trial_ticks=TRIAL_TICKS,
                    pairs=MODULARITY_PAIRS):
    """One generation: mutate a batch from the parent, score each on ALL pairs, return sorted (score, creature)."""
    tuner = Finetune(seed_creature, batch_size=batch_size)
    scored = [
        (evaluate_multi(c, trial_ticks, pairs), c)
        for c in tuner.finetunes
    ]
    scored.append((evaluate_multi(seed_creature, trial_ticks, pairs), seed_creature))

    scored.sort(key=lambda x: (x[0], -len(x[1].all_synapses)), reverse=True) # sort by scores first, then the lessity of synapses (so it removes redundancy in the long run)
    return scored


def run_neat(seed_creature, generations=GENERATIONS, batch_size=BATCH_SIZE, trial_ticks=TRIAL_TICKS,
             pairs=MODULARITY_PAIRS, on_generation=None):
    """Runs the full generation loop. `on_generation(gen, best_score, best_creature)` is an
    optional callback, e.g. to update the pygame display between generations."""
    best = seed_creature
    history = []
    for gen in range(generations):
        scored = run_generation(best, batch_size, trial_ticks, pairs)
        best_score, best = scored[0]
        history.append(best_score)
        print(f"gen {gen:3d}: best fitness = {best_score:+.2f}  (max possible = {len(pairs):+.2f})")
        if on_generation is not None:
            on_generation(gen, best_score, best, scored)
    return best, history


def health_report():
    dead = [i for i in range(len(neurons)) if len(fire_history[i]) >= 50 and firing_rate(i) == 0.0]
    stuck = [i for i in range(len(neurons)) if len(fire_history[i]) >= 50 and firing_rate(i) > 0.9]
    return dead, stuck


def firing_rate(i):
    h = fire_history[i]
    return sum(h) / len(h) if h else 0.0


def diagnostics(creature, neurons, output_neurons, made, attempts, n_synapses):
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
    print(f"\nwired: {made}/{n_synapses} synapses after {attempts} attempts\n")

    for i, n in enumerate(neurons):
        outs = [neurons.index(s.receiver) for s in n.outgoing_synapses]
        print(f"n{i} ({'INH' if getattr(n, 'is_inhibitory', False) else 'exc'}) -> {outs}")


# =============== pygame setup (comes up FIRST so evolution can be watched live) ===============
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("window")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 14)
small = pygame.font.SysFont("consolas", 12)

PREVIEW_TICKS = 90  # how many ticks of each generation's best creature to animate before moving on


class QuitRequested(Exception):
    """Raised when the window is closed / ESC is hit mid-evolution, so we can bail out cleanly."""


# ---------------- shared playback state ----------------
population = []
pop_scores = []
current_idx = 0
creature = neurons = input_neurons = output_neurons = None
tick = 0
paused = False
fire_history = []
flash = []
event_log = []
gen_status = ""
running = True
fast_forward = False
skip_gen_preview = False
y = 0


def load_creature(idx):
    """Swap in population[idx] as the creature being visualized, resetting all playback state."""
    global creature, neurons, input_neurons, output_neurons
    global current_idx, tick, fire_history, flash, event_log

    current_idx = idx % len(population)
    creature = population[current_idx]
    neurons = creature.neurons
    input_neurons = [n for n in neurons if n.is_input_neuron]
    output_neurons = [n for n in neurons if n.is_output_neuron]
    reset_creature(creature)

    tick = 0
    fire_history = [[] for _ in neurons]
    flash = [0] * len(neurons)
    event_log = []
    log(f"loaded creature {current_idx}/{len(population) - 1}  score {pop_scores[current_idx]:+.2f}")


def brain_tick(kick=False):
    """One full creature step, plus a manual kick on an input neuron."""
    global tick

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


def handle_events():
    """Pump the pygame event queue and apply whatever key/quit the user made."""
    global running, paused, TICKS_PER_SECOND, skip_gen_preview, fast_forward
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
            elif event.key == pygame.K_RIGHTBRACKET:
                load_creature(current_idx + 1)
            elif event.key == pygame.K_LEFTBRACKET:
                load_creature(current_idx - 1)
            elif event.key == pygame.K_RETURN:
                skip_gen_preview = True   # jump straight to the next generation
            elif event.key == pygame.K_TAB:
                fast_forward = True       # stop animating, just flash through to the final batch


def text(msg, color=(220, 220, 230), f=font):
    global y
    screen.blit(f.render(msg, True, color), (WORLD_W + 10, y))
    y += 18


def draw():
    global y
    screen.fill((8, 8, 14))

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

    for i, n in enumerate(creature.neurons):
        firing = flash[i] > 0
        is_io = n.is_input_neuron or n.is_output_neuron
        radius = (10 if is_io else 8) + (3 if firing else 0)
        base = (255, 255, 255) if firing else n.color

        pygame.draw.circle(screen, base, n.pos, radius)
        if is_io:
            pygame.draw.circle(screen, (255, 255, 255), n.pos, radius, 2)

        pot = max(0.0, min(n.potential / max(n.threshold, 1e-6), 1.0))
        pygame.draw.circle(screen, (90, 255, 150), n.pos, radius + 4, 1 + int(pot * 3))

        label_txt = (f"I{i}" if n.is_input_neuron else f"O{i}" if n.is_output_neuron else str(i))
        label = small.render(label_txt, True, (230, 230, 230) if is_io else (200, 200, 210))
        label_x = n.pos[0] - 24 if n.is_input_neuron else n.pos[0] + 12
        screen.blit(label, (label_x, n.pos[1] - 7))

    text_h = font.render("INPUTS", True, INPUT_COLOR)
    screen.blit(text_h, (MARGIN - 30, 6))
    text_o = font.render("OUTPUTS", True, OUTPUT_COLOR)
    screen.blit(text_o, (WORLD_W - MARGIN - 30, 6))

    pygame.draw.rect(screen, (16, 16, 26), (WORLD_W, 0, PANEL_W, SCREEN_H))
    y = 8

    text(gen_status, (255, 255, 140))
    text(f"creature {current_idx}/{len(population) - 1}   score {pop_scores[current_idx]:+.2f}",
         (255, 255, 140))
    text(f"tick {tick}   {'PAUSED' if paused else 'running'}")
    text(f"speed {TICKS_PER_SECOND} ticks/s")
    text("[ / ] prev / next creature", (140, 220, 220), small)
    text("ENTER skip gen  TAB skip to end", (140, 220, 220), small)
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


def on_generation(gen, best_score, best, scored):
    """Called by run_neat after every generation: load that generation's best creature
    and animate it for a bit, so evolution is actually visible on screen as it happens."""
    global population, pop_scores, gen_status, skip_gen_preview

    handle_events()
    if not running:
        raise QuitRequested()

    population = [c for _, c in scored]
    pop_scores = [s for s, _ in scored]
    gen_status = f"evolving   gen {gen + 1}/{GENERATIONS}   best {best_score:+.2f}"
    load_creature(0)

    if fast_forward:
        draw()
        return

    skip_gen_preview = False
    for _ in range(PREVIEW_TICKS):
        handle_events()
        if not running:
            raise QuitRequested()
        if skip_gen_preview:
            break
        brain_tick()
        draw()
        clock.tick(60)


# =============== build the seed creature and run NEAT, watching it evolve live ===============
creature, neurons, input_neurons, output_neurons, made, attempts = make_sane_creature(
    rng, N_INPUT, N_OUTPUT, N_HIDDEN, MARGIN, WORLD_W, SCREEN_H,
    INPUT_COLOR, OUTPUT_COLOR, HIDDEN_COLOR, N_SYNAPSES
)

diagnostics(creature, neurons, output_neurons, made, attempts, N_SYNAPSES)

print("\n--- running NEAT (synaptic tuning only) ---")
try:
    best_creature, fitness_history = run_neat(creature, generations=GENERATIONS, batch_size=BATCH_SIZE,
                                               trial_ticks=TRIAL_TICKS, on_generation=on_generation)
except QuitRequested:
    pygame.quit()
    sys.exit()

print(f"\nfinal best fitness: {fitness_history[-1]:+.2f}")
print(f"fitness history: {fitness_history}")

# build one more scored batch off the evolved best, so we have a whole
# population of creatures to click through and visually debug
final_batch = run_generation(best_creature, batch_size=BATCH_SIZE, trial_ticks=TRIAL_TICKS)
population = [best_creature] + [c for _, c in final_batch]
pop_scores = [fitness_history[-1]] + [s for s, _ in final_batch]
gen_status = "evolution done -- browsing final batch"
load_creature(0)

# ---------------- interactive playback of the final batch ----------------
tick_timer = 0.0

while running:
    dt = clock.tick(60) / 1000.0
    handle_events()

    if not paused:
        tick_timer += dt
        step = 1.0 / TICKS_PER_SECOND
        while tick_timer >= step:
            tick_timer -= step
            kick = KICK_EVERY > 0 and tick % KICK_EVERY == 0
            brain_tick(kick=kick)
            if tick % 300 == 0:
                reset_creature(creature)  # loop the demo trial so you can watch it repeatedly

    draw()

pygame.quit()
sys.exit()