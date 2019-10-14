from abc import ABC, abstractmethod


class Evaluator(ABC):
    C1 = 1.0
    C2 = 1.0
    C3 = 0.4

    def __init__(self, population_size, starting_genome, node_innovation, con_innovation):
        self.population_size = population_size
        self.node_innovation = node_innovation
        self.con_innovation = con_innovation
        self.genomes = self.list_of_stating_genomes(starting_genome)
        self.next_generation = []
        self.genome_species = {}
        self.genome_fitness = {}
        self.species = []
        self.highest_score = 0
        self.fittest_genome = starting_genome
        super().__init__()

    def list_of_stating_genomes(self, starting_genome):
        list_of_genomes = []
        for i in range(self.population_size):
            list_of_genomes.append(starting_genome)
        return list_of_genomes

    def evaluate(self):
        pass

    def get_random_genome_biased_adjusted_fitness(self):
        pass

    @abstractmethod
    def evaluate_genome(self):
        pass

    class FitnessGenome:
        def __init__(self, genome, fitness):
            self.genome = genome
            self.fitness = fitness

    class Species:
        def __init__(self, mascot):
            self.mascot = mascot
            self.members = [self.mascot]
            self.fitness_population = []
            self.total_adjusted_fitness = 0.0
