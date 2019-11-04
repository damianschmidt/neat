from NEAT_new.src.innovation import InnovationSet


class GeneticAlgorithm:
    def __init__(self, task):
        self.genomes = []
        self.species = []
        self.bests = []
        self.best_ever = None
        self.innovation_set = InnovationSet()
        self.task = task
        self.generation = 0

        # parameters
        self.population_size = 100
        self.number_generation_allowed_to_not_improve = 20
        self.crossover_rate = 0.7
