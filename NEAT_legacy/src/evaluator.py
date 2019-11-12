import operator
import random
from abc import ABC, abstractmethod

from NEAT_legacy.src import genome_utils
from NEAT_legacy.src.fitness_genome import FitnessGenome
from NEAT_legacy.src.genome import Genome
from NEAT_legacy.src.neat_conf import Config
from NEAT_legacy.src.species import Species


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
        self.fitness_genomes = []
        self.highest_score = 0
        self.fittest_genome = starting_genome
        self.stagnation = 0
        super().__init__()

    def list_of_starting_genomes(self, starting_genome):
        list_of_genomes = []
        for i in range(self.population_size):
            list_of_genomes.append(starting_genome)
        return list_of_genomes

    def evaluate(self):
        self.stagnation += 1
        self.reset_before_generation()
        self.place_genomes_into_species()
        self.remove_species_without_genomes()
        self.evaluate_genomes_and_assign_fitness()
        self.best_into_next_generation()
        self.remove_stagnation()
        self.breed_rest_of_genomes()
        self.genomes = self.next_generation

    def reset_before_generation(self):
        for s in self.species:
            s.reset()
        self.genome_fitness = {}
        self.genome_species = {}
        self.next_generation = []
        self.fitness_genomes = []
        self.highest_score = 0
        self.fittest_genome = None

    def place_genomes_into_species(self):
        for genome in self.genomes:
            found_species = False
            for s in self.species:
                # if compatibility distance is less than species threshold then genome belongs to species
                if genome_utils.compatibility_distance(genome, s.mascot, Config.C1, Config.C2,
                                                       Config.C3) < Config.SPECIES_THRESHOLD:
                    s.members.append(genome)
                    self.genome_species[genome] = s
                    found_species = True
                    break

            if not found_species:  # if there is no species applied for genome, create new species
                new_species = Species(genome)
                self.species.append(new_species)
                self.genome_species[genome] = new_species

    def remove_species_without_genomes(self):
        self.species = [s for s in self.species if len(s.members) != 0]

    def evaluate_genomes_and_assign_fitness(self):
        for genome in self.genomes:
            s = self.genome_species[genome]
            score = self.evaluate_genome(genome)
            adjust_score = score / len(self.genome_species[genome].members)
            fitness_genome = FitnessGenome(genome, adjust_score)
            s.total_adjusted_fitness += adjust_score
            s.fitness_population.append(fitness_genome)
            self.fitness_genomes.append(fitness_genome)
            self.genome_fitness[genome] = score
            if score > self.highest_score:
                self.stagnation = 0
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

    def remove_stagnation(self):
        if self.stagnation > Config.STAGNATION:
            self.fitness_genomes = sorted(self.fitness_genomes, key=operator.attrgetter('fitness'), reverse=True)
            for i in range(5, len(self.fitness_genomes)):
                fitness_genome = self.fitness_genomes[i]
                self.genomes.remove(fitness_genome.genome)
                for species in self.species:
                    species.members.remove(fitness_genome.genome)
                    species.fitness_population.remove(fitness_genome)
                    del self.genome_species[fitness_genome.genome]
                    del self.genome_fitness[fitness_genome.genome]
            self.remove_species_without_genomes()
            self.stagnation = 0

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
            if random.random() < Config.ADD_NODE_RATE:
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
