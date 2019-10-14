import random
from abc import ABC, abstractmethod

from NEAT.src import utils


class Evaluator(ABC):
    C1 = 1.0
    C2 = 1.0
    C3 = 0.4
    SPECIES_THRESHOLD = 10.0

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
        self.reset_before_generation()
        self.place_genomes_into_species()
        self.remove_species_without_genomes()
        self.evaluate_genomes_and_assign_fitness()
        self.best_into_next_generation()
        # Breed the rest of the genomes

    def best_into_next_generation(self):
        for s in self.species:
            reversed_fitness_population = s.fitness_population.sort(reverse=True)
            fittest_in_species = reversed_fitness_population[0]
            self.next_generation.append(fittest_in_species)

    def evaluate_genomes_and_assign_fitness(self):
        for genome in self.genomes:
            s = self.genome_species[genome]
            score = self.evaluate_genome(genome)
            adjust_score = score / len(self.genome_species[genome].members)
            s.total_adjusted_fitness += adjust_score
            self.fittest_genome[genome] = score
            if score > self.highest_score:
                self.highest_score = score
                self.fittest_genome = genome

    def remove_species_without_genomes(self):
        self.species = [s for s in self.species if len(s.members) != 0]

    def place_genomes_into_species(self):
        for genome in self.genomes:
            found_species = False
            for s in self.species:
                # if compatibility distance is less than species threshold then genome belongs to species
                if utils.compatibility_distance(genome, s.mascot, self.C1, self.C2, self.C3) < self.SPECIES_THRESHOLD:
                    s.members.append(genome)
                    self.genome_species[genome] = s
                    found_species = True
                    break

                if not found_species:  # if there is no species applied for genome, create new species
                    new_species = self.Species(genome)
                    self.species.append(new_species)
                    self.genome_species[genome] = new_species

    def reset_before_generation(self):
        for s in self.species:
            s.reset()
        self.genome_fitness = {}
        self.genome_species = {}
        self.next_generation = []
        self.highest_score = 0
        self.fittest_genome = None

    def get_random_genome_biased_adjusted_fitness(self):
        pass

    @abstractmethod
    def evaluate_genome(self, genome):
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

        def reset(self):
            new_mascot_index = random.randint(0, len(self.members) - 1)
            self.mascot = self.members[new_mascot_index]
            self.members = [self.mascot]
            self.fitness_population = []
            self.total_adjusted_fitness = 0.0
