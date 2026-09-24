from neuron import Neuron, Synapse
import copy
class Creature:
    def __init__(self, 
                 neurons: list[Neuron] = []
                 ):
        self.neurons = neurons
        #self.all_synapses = [s for n in neurons for s in n.outgoing_synapses]
        self.input_neurons = [n for n in neurons if n.is_input_neuron]
        self.output_neurons = [n for n in neurons if n.is_output_neuron]
        self.ticks = 0

    @property
    def all_synapses(self) -> list[Synapse]:
        """Dynamically pulls the up-to-date active synapses across the graph structure."""
        return [s for n in self.neurons for s in n.outgoing_synapses]
    
    def clone(self) -> 'Creature':
        """Creates a perfect structural duplicate of the creature, preserving cyclic graph wiring."""
        memo = {}
        # copy.deepcopy automatically manages cross-references inside the graph using the memo dict
        cloned_neurons = copy.deepcopy(self.neurons, memo)
        
        cloned_creature = Creature(cloned_neurons)
        cloned_creature.ticks = self.ticks
        return cloned_creature
    
    def brain_tick(self):
        
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
    

