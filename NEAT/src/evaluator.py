import random
import operator
from abc import ABC, abstractmethod

from NEAT.src import genome_utils
from NEAT.src.genome import Genome
from NEAT.src.neat_conf import Config


class Evaluator(ABC):
    def __init__(self, starting_genome, node_innovation, con_innovation):
        self.population_size = Config.POPULATION_SIZE
        self.node_innovation = node_innovation
        self.con_innovation = con_innovation
        self.genomes = self.list_of_starting_genomes(starting_genome)
        self.next_generation = []
        self.genome_species = {}
        self.genome_fitness = {}
        self.species = []
        self.highest_score = 0
        self.fittest_genome = starting_genome
        super().__init__()

    def list_of_starting_genomes(self, starting_genome):
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
        self.breed_rest_of_genomes()
        self.genomes = self.next_generation

    def reset_before_generation(self):
        for s in self.species:
            s.reset()
        self.genome_fitness = {}
        self.genome_species = {}
        self.next_generation = []
        self.highest_score = 0
        self.fittest_genome = None

    def place_genomes_into_species(self):
        species = self.species.copy()
        for genome in self.genomes:
            found_species = False
            for s in species:
                # if compatibility distance is less than species threshold then genome belongs to species
                if genome_utils.compatibility_distance(genome, s.mascot, Config.C1, Config.C2,
                                                       Config.C3) < Config.SPECIES_THRESHOLD:
                    s.members.append(genome)
                    self.genome_species[genome] = s
                    found_species = True
                    break

                if not found_species:  # if there is no species applied for genome, create new species
                    new_species = self.Species(genome)
                    self.species.append(new_species)
                    self.genome_species[genome] = new_species

    def remove_species_without_genomes(self):
        self.species = [s for s in self.species if len(s.members) != 0]

    def evaluate_genomes_and_assign_fitness(self):
        for genome in self.genomes:
            s = self.genome_species[genome]
            score = self.evaluate_genome(genome)
            adjust_score = score / len(self.genome_species[genome].members)
            fitness_genome = Evaluator.FitnessGenome(genome, adjust_score)
            s.total_adjusted_fitness += adjust_score
            s.fitness_population.append(fitness_genome)
            self.genome_fitness[genome] = score
            if score > self.highest_score:
                self.highest_score = score
                self.fittest_genome = genome

    @abstractmethod
    def evaluate_genome(self, genome):
        pass

    def best_into_next_generation(self):
        for s in self.species:
            reversed_fitness_population = sorted(s.fitness_population, key=operator.attrgetter('fitness'), reverse=True)
            fittest_in_species = reversed_fitness_population[0]
            self.next_generation.append(fittest_in_species.genome)

    def breed_rest_of_genomes(self):
        while len(self.next_generation) < self.population_size:
            species = self.get_random_species_biased_adjusted_fitness()
            genome_parent1 = self.get_random_genome_biased_adjusted_fitness(species)
            genome_parent2 = self.get_random_genome_biased_adjusted_fitness(species)
            child = Genome()
            if self.genome_fitness[genome_parent1] >= self.genome_fitness[genome_parent2]:
                child = child.crossover(genome_parent1, genome_parent2)
            else:
                child = child.crossover(genome_parent2, genome_parent1)

            if random.random() < Config.MUTATION_RATE:
                child.mutation()
            if random.random() < Config.ADD_CONNECTION_RATE:
                child.add_connection_mutation()
            if random.random < Config.ADD_NODE_RATE:
                child.add_node_mutation()
            self.next_generation.append(child)

    def get_random_species_biased_adjusted_fitness(self):
        complete_fitness_of_species = 0.0
        for s in self.species:
            complete_fitness_of_species += s.total_adjusted_fitness

        random_value = random.random() * complete_fitness_of_species
        current_fitness = 0.0
        for s in self.species:
            current_fitness += s.total_adjusted_fitness
            if current_fitness >= random_value:
                return s

    def get_random_genome_biased_adjusted_fitness(self, species):
        complete_fitness_of_fitness_genomes = 0.0
        for fitness_genome in species.fitness_population:
            complete_fitness_of_fitness_genomes += fitness_genome.fitness

        random_value = random.random() * complete_fitness_of_fitness_genomes
        current_fitness = 0.0
        for fitness_genome in species.fitness_population:
            current_fitness += fitness_genome.fitness
            if current_fitness >= random_value:
                return fitness_genome.genome

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
