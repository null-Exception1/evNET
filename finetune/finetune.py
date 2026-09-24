from creature import Creature
class Finetune:
    def __init__(self, 
                 creature: Creature,
                 batch_size: int
                 ):
        self.original_creature = creature
        self.finetunes: list[Creature] = []
        self.batch_size: int = batch_size

        self.create_finetunes()

    def create_finetunes(self):
        # will add some more logic later
        self.finetunes = []
        for i in range(self.batch_size):
            
            # do some tuning in synapses...
            final_creature = self.original_creature

            # add 
            self.finetunes.append(final_creature)