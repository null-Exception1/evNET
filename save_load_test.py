
import random
from neuron import Neuron
from creature import Creature


def build_creature(n_neurons=10, n_inputs=2, n_outputs=2):
    neurons = []
    for i in range(n_neurons):
        neurons.append(Neuron(
            pos=(random.uniform(0, 300), random.uniform(0, 300)),
            color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
            is_input_neuron=i < n_inputs,
            is_output_neuron=i >= n_neurons - n_outputs,
            is_inhibitory=(i % 4 == 3),
        ))
    for a in neurons:
        for b in random.sample(neurons, 4):
            if a.is_output_neuron or (a.is_input_neuron and b.is_input_neuron):
                continue
            w = random.uniform(0.4, 1.2)
            a.add_synapse(b, -w if a.is_inhibitory else w, delay=random.randint(1, 4))
    return Creature(neurons)


def run(creature, ticks=30):
    history = []
    for t in range(ticks):
        for n in creature.get_input_neurons():
            n.fired = (t % 3 == 0)
        creature.brain_tick()
        history.append([n.fired for n in creature.neurons])
    return history


if __name__ == "__main__":
    original = build_creature()
    original.save("my_creature.json")
    print(f"saved: {len(original.neurons)} neurons, {len(original.all_synapses)} synapses")

    loaded = Creature.load("my_creature.json")
    print(f"loaded: {len(loaded.neurons)} neurons, {len(loaded.all_synapses)} synapses")

    same = run(original) == run(loaded)
    print("same behaviour after loading:", same)

  
    copy_of_loaded = loaded.clone()
    loaded.save("my_creature_resaved.json") 