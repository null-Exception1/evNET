# synapse.py
from __future__ import annotations
import numpy as np


class Synapse:
    def __init__(self, weight: float, sender, receiver, delay: int = 1):
        assert delay >= 1, "delay must be at least 1 tick (0 would make results order-dependent)"
        self.weight = weight
        self.sender = sender
        self.receiver = receiver
        self.delay = delay
        self.eligibility = 0.0

        # buffer[k] = 1 means a spike arrives in k ticks; buffer[0] arrives now
        self.buffer = np.zeros(delay, dtype=bool)
        self.spike = False          # did a spike arrive at the receiver THIS tick?

    def send(self, fired: bool):
        """Sender's decision this tick: schedule it to land `delay` ticks from now."""
        self.buffer[-1] = fired

    def advance(self):
        """Move the buffer forward one tick; whatever reaches the front is the arriving spike."""
        self.spike = bool(self.buffer[0])
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = False

    def in_flight(self):
      """Fractions (0..1) along the edge for each spike still travelling.
      0 = just left the sender, approaching 1 = about to land on the receiver."""
      return [1.0 - (k + 1) / self.delay for k in np.flatnonzero(self.buffer)]
