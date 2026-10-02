from creature import Creature
import random 
import math
import copy
class Finetune:
    def __init__(self, 
                 creature: Creature,
                 batch_size: int
                 ):
        self.original_creature = creature
        self.finetunes: list[Creature] = []
        self.batch_size: int = batch_size

        self.randomize_synapse_weights = 0.2
        self.randomize_neuron_wiring = 0.7
        self.create_finetunes()
    @staticmethod
    def delay_from_distance(a, b):
        VELOCITY = 70
        return max(1, round(math.dist(a.pos, b.pos) / VELOCITY))

    def create_finetunes(self):
        # will add some more logic later
        self.finetunes = []
        for i in range(self.batch_size):
            
            # do some tuning in synapses.. to like 20% of the connections (adjustable)
            
            final_creature = self.original_creature.clone()
            for synapse in final_creature.all_synapses:
                if random.random() < self.randomize_synapse_weights: # 20% chance of changing up the weights
                    raw_weight = float(random.uniform(0.4, 1.2))
                    synapse.weight = -raw_weight if synapse.sender.is_inhibitory else raw_weight

            
            
            for a in final_creature.neurons: # right now only 70%
                if random.random() < self.randomize_neuron_wiring:
                    if len(a.outgoing_synapses) >= 1 and random.random() < 0.5:
                        a.delete_synapse(random.choice(a.outgoing_synapses).receiver) 
                    else:
                        b = random.choice(final_creature.neurons)
                        raw_weight = float(random.uniform(0.4, 1.2))   # magnitude only, always positive draw
                        weight = -raw_weight if a.is_inhibitory else raw_weight
                        
                        if not ((a.is_input_neuron and b.is_output_neuron) or (a.is_output_neuron and b.is_input_neuron) or (a.is_input_neuron and b.is_input_neuron)  or (a.is_output_neuron and b.is_output_neuron)):
                            a.add_synapse(
                                b,
                                weight,
                                delay=self.delay_from_distance(a, b),
                            )
                    
            # add 
            self.finetunes.append(final_creature)