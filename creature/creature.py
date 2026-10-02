from neuron import Neuron, Synapse
import copy
import json
import math
import numpy as np
import random

FORMAT_VERSION = 1

class Creature:
    def __init__(self, 
                 neurons: list[Neuron] = []
                 ):
        self.neurons = neurons
        #self.all_synapses = [s for n in neurons for s in n.outgoing_synapses]
        self.input_neurons = [n for n in neurons if n.is_input_neuron]
        self.output_neurons = [n for n in neurons if n.is_output_neuron]
        self.ticks = 0
        self.chem_decay = 0.7 # manual for now
        self.chem_range = 120
        self.chem_amp= 5
    @property
    def all_synapses(self) -> list[Synapse]:
        return [s for n in self.neurons for s in n.outgoing_synapses]
    
    def clone(self) -> 'Creature':
        memo = {}
        cloned_neurons = copy.deepcopy(self.neurons, memo)
        
        cloned_creature = Creature(cloned_neurons)
        cloned_creature.ticks = self.ticks
        return cloned_creature

    def chem_tick(self, neuron: Neuron):

        # some chemistry shit going on here

        neuron.chem_inputs *= self.chem_decay
        #print(neuron.chem_inputs)
        for n2 in self.neurons:
            if not n2.chem_release.any(): continue
            if id(neuron) != id(n2):
                distance = math.hypot(neuron.pos[0]-n2.pos[0],neuron.pos[1]-n2.pos[1])
                #print("chem inputs: ",n2.chem_release,", self ticks",self.ticks)
                neuron.chem_inputs += n2.chem_release * np.exp(-(distance**1) / self.chem_range) * self.chem_amp
            
        neuron.chem_inputs = np.clip(neuron.chem_inputs,0,5)

    def brain_tick(self):
        for n in self.neurons:
            self.chem_tick(n)
        
        for n in self.neurons:
            n.process()
        for n in self.neurons:
            n.forward()
        for s in self.all_synapses:
            s.advance()

        self.ticks += 1
    def get_input_neurons(self):
        return self.input_neurons
    def read_outputs(self) -> list[bool]:
        return [n.fired for n in self.output_neurons]

    
    @staticmethod
    def _arr(a):
        return np.asarray(a, dtype=float).tolist()

    def to_dict(self) -> dict:
        """Structure only (wiring, weights, delays, chem params) as a JSON-serialisable dict."""
        neurons = self.neurons
        index_of = {id(n): i for i, n in enumerate(neurons)}

        # give every synapse an id (collect from both lists so none are missed)
        syn_id = {}
        syn_list = []
        for n in neurons:
            for s in list(n.outgoing_synapses) + list(n.incoming_synapses):
                if id(s) not in syn_id:
                    syn_id[id(s)] = len(syn_list)
                    syn_list.append(s)

        synapses = [
            {
                "weight": float(s.weight),
                "sender": index_of[id(s.sender)],
                "receiver": index_of[id(s.receiver)],
                "delay": int(s.delay),
            }
            for s in syn_list
        ]

        neuron_data = []
        for n in neurons:
            neuron_data.append({
                "pos": list(map(float, n.pos)),
                "color": list(n.color) if not isinstance(n.color, str) else n.color,
                "is_input_neuron": bool(n.is_input_neuron),
                "is_output_neuron": bool(n.is_output_neuron),
                "is_inhibitory": bool(n.is_inhibitory),
                # hyperparameters
                "target_rest": float(n.target_rest),
                "threshold_margin": float(n.threshold_margin),
                "potential_leak": float(n.potential_leak),
                "spike_leak": float(n.spike_leak),
                "target_rate": float(n.target_rate),
                "homeostasis_tau": float(n.homeostasis_tau),
                "number_of_input_chemicals": int(n.number_of_input_chemicals),
                "number_of_output_chemicals": int(n.number_of_output_chemicals),
                # FFN
                "W1": self._arr(n.W1), "B1": self._arr(n.B1),
                "W2": self._arr(n.W2), "B2": self._arr(n.B2),
                # connectivity (indices into "synapses", order preserved)
                "outgoing": [syn_id[id(s)] for s in n.outgoing_synapses],
                "incoming": [syn_id[id(s)] for s in n.incoming_synapses],
            })

        return {
            "format_version": FORMAT_VERSION,
            "creature": {
                "chem_decay": float(self.chem_decay),
                "chem_range": float(self.chem_range),
                "chem_amp": float(self.chem_amp),
            },
            "neurons": neuron_data,
            "synapses": synapses,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Creature':
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported save format: {data.get('format_version')}")

        neurons = []
        for d in data["neurons"]:
            color = tuple(d["color"]) if isinstance(d["color"], list) else d["color"]
            n = Neuron(
                pos=tuple(d["pos"]),
                color=color,
                target_rest=d["target_rest"],
                threshold_margin=d["threshold_margin"],
                potential_leak=d["potential_leak"],
                spike_leak=d["spike_leak"],
                number_of_input_chemicals=d["number_of_input_chemicals"],
                number_of_output_chemicals=d["number_of_output_chemicals"],
                hidden_size=len(d["B1"]),
                target_rate=d["target_rate"],
                homeostasis_tau=d["homeostasis_tau"],
                is_input_neuron=d["is_input_neuron"],
                is_output_neuron=d["is_output_neuron"],
                is_inhibitory=d["is_inhibitory"],
            )
            # overwrite the randomised FFN, then re-derive thresholds from it
            n.W1 = np.array(d["W1"]); n.B1 = np.array(d["B1"])
            n.W2 = np.array(d["W2"]); n.B2 = np.array(d["B2"])
            n.recalibrate()
            n.min_threshold = n._fire_signal(0.0, 1.0) + 0.02
            n.max_threshold = n._fire_signal(3.0, 0.0) - 0.02
            n.threshold = float(np.clip(n.threshold, n.min_threshold, n.max_threshold))
            n.incoming_synapses = []
            n.outgoing_synapses = []
            neurons.append(n)

        # build synapses directly (bypassing add_synapse's guards) so the
        # graph is restored exactly as saved
        synapses = []
        for sd in data["synapses"]:
            s = Synapse(sd["weight"], neurons[sd["sender"]], neurons[sd["receiver"]], sd["delay"])
            synapses.append(s)

        for n, d in zip(neurons, data["neurons"]):
            n.outgoing_synapses = [synapses[i] for i in d["outgoing"]]
            n.incoming_synapses = [synapses[i] for i in d["incoming"]]

        c = cls(neurons)
        cd = data["creature"]
        c.chem_decay = cd["chem_decay"]
        c.chem_range = cd["chem_range"]
        c.chem_amp = cd["chem_amp"]
        return c

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> 'Creature':
        with open(path) as f:
            return cls.from_dict(json.load(f))