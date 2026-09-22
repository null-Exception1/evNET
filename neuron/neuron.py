# neuron.py
from __future__ import annotations
from .synapse import Synapse
import numpy as np


class Neuron:
    def __init__(
        self,
        pos,
        color,
        rng: np.random.Generator | None = None,
        target_rest: float = 0.2,
        threshold_margin: float = 0.15,
        input_gain: float = 1.0,
        potential_leak: float = 0.2,
        spike_leak: float = 0.2,
        number_of_input_chemicals: int = 4,
        number_of_output_chemicals: int = 4,
        hidden_size: int = 4,
    ):
        self.pos = pos
        self.color = color
        self.rng = rng if rng is not None else np.random.default_rng()

        # connections are only ever added through add_synapse, so the
        # two lists can't drift out of sync
        self.incoming_synapses: list[Synapse] = []
        self.outgoing_synapses: list[Synapse] = []

        self.potential = 0.0
        self.potential_leak = potential_leak
        self.spike_trace = 0.0
        self.spike_leak = spike_leak

        self.number_of_input_chemicals = number_of_input_chemicals
        self.number_of_output_chemicals = number_of_output_chemicals

        # FFN input:  [potential, spike_trace, chem_1..chem_n]
        # FFN output: [fire_decision, release_1..release_m]
        self.n_in = 2 + number_of_input_chemicals
        self.n_out = 1 + number_of_output_chemicals

        self.W1 = self.rng.standard_normal((self.n_in, hidden_size))
        self.B1 = self.rng.standard_normal(hidden_size)
        self.W2 = self.rng.standard_normal((hidden_size, self.n_out))
        self.B2 = self.rng.standard_normal(self.n_out)

        # --- make firing respond UPWARD to the potential ---
        # potential -> hidden weights positive (and amplified), hidden -> fire output positive.
        # sigmoid is monotonic, so higher potential can now only raise the fire signal.
        self.W1[0, :] = np.abs(self.W1[0, :]) * input_gain
        self.W2[:, 0] = np.abs(self.W2[:, 0])

        # --- per-neuron operating point: rest at target_rest, fire at rest + margin ---
        self.target_rest = target_rest
        self.threshold_margin = threshold_margin
        self.threshold = 0.0        # set by recalibrate()
        self.recalibrate()

        self.fired = False
        self.chem_inputs = np.zeros(number_of_input_chemicals)
        self.chem_release = np.zeros(number_of_output_chemicals)
        self.incoming = []
        self.last_fire_signal = 0.0   # raw a2[0], handy for debugging

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _forward(self, x):
        """Run the FFN on an input vector, return the full output vector."""
        a1 = self.sigmoid(x @ self.W1 + self.B1)
        return self.sigmoid(a1 @ self.W2 + self.B2)

    def _fire_signal(self, potential, spike_trace, chem=None):
        """Fire output for a hypothetical input. Used for calibration and diagnostics."""
        chem = np.zeros(self.number_of_input_chemicals) if chem is None else chem
        x = np.concatenate(([potential, spike_trace], chem))
        return float(self._forward(x)[0])

    def recalibrate(self):
        """Pin the resting fire signal at target_rest, then set the threshold just above it.
        Call again after anything that changes the FFN weights or biases."""
        eps = 1e-6
        rest_now = float(np.clip(self._fire_signal(0.0, 0.0), eps, 1 - eps))
        target = float(np.clip(self.target_rest, eps, 1 - eps))

        # sigmoid is monotonic, so shifting the pre-sigmoid bias by the logit
        # difference moves the resting output to exactly `target`
        logit = lambda p: np.log(p / (1 - p))
        self.B2[0] += logit(target) - logit(rest_now)

        self.threshold = target + self.threshold_margin

    def sensitivity_report(self, high_potential: float = 3.0):
        """Does this neuron respond upward to input, and can input cross its threshold?"""
        rest = self._fire_signal(0.0, 0.0)
        high = self._fire_signal(high_potential, 0.0)
        return {
            "rest": rest,
            "high": high,
            "threshold": self.threshold,
            "silent_at_rest": rest <= self.threshold,
            "can_fire": high > self.threshold,
        }

    def add_synapse(self, target: Neuron, weight: float, delay: int = 1):
      if target is self:
          return
      if any(s.receiver is target for s in self.outgoing_synapses):
          return
      s = Synapse(weight, self, target, delay)
      self.outgoing_synapses.append(s)
      target.incoming_synapses.append(s)

    def delete_synapse(self, target: Neuron):
        doomed = [s for s in self.outgoing_synapses if s.receiver is target]
        self.outgoing_synapses = [
            s for s in self.outgoing_synapses if s.receiver is not target
        ]
        for s in doomed:
            target.incoming_synapses.remove(s)   # remove from BOTH lists

    def push_synapse_inputs_to_neuron(self):
        self.incoming = [s.weight if s.spike else 0.0 for s in self.incoming_synapses]

    def input_pass(self):
        drive = sum(self.incoming)
        self.potential = self.potential * self.potential_leak + drive

        noise = self.rng.normal(0.0, 0.05, size=self.chem_inputs.shape)
        noisy_chem = self.chem_inputs + noise

        x = np.concatenate(([self.potential, self.spike_trace], noisy_chem))
        a2 = self._forward(x)
        self.last_fire_signal = float(a2[0])

        fired = bool(a2[0] > self.threshold)
        if fired:
            self.potential = 0.0
        self.spike_trace = self.spike_trace * self.spike_leak + float(fired)

        release = a2[1:] if fired else np.zeros_like(a2[1:])
        return fired, release

    def process(self):
        """Pass 1: read last tick's spike flags and decide. Writes nothing to synapses."""
        self.push_synapse_inputs_to_neuron()
        self.fired, self.chem_release = self.input_pass()

    def forward(self):
      """Pass 2: hand this tick's spike to every outgoing synapse."""
      for s in self.outgoing_synapses:
          s.send(self.fired)