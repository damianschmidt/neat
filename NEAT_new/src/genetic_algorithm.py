import time
from statistics import mean, stdev

from NEAT_new.src.reproduction import Reproduction
from NEAT_new.src.species import SpeciesSet
from NEAT_new.src.stagnation import Stagnation


class GeneticAlgorithm:
    def __init__(self, config):
        self.generation = 0
        self.best = None
        self.config = config
        self.stagnation = Stagnation(config)
        self.reproduction = Reproduction(self.stagnation, config)
        self.population = self.reproduction.create_new(config.population_size)
        self.species = SpeciesSet()
        self.species.speciate(config, self.population, self.generation)
        self.generation_start = None
        self.generation_times = []

    def run(self, fitness_function, epochs=None):
        i = 0
        while epochs is None or i < epochs:
            i += 1

            print(f'\n ++++++++++ GENERATION {self.generation} ++++++++++ \n')
            self.generation_start = time.time()

            # evaluate all genomes
            fitness_function(list(self.population.items()))

            # get best and report
            best = None
            for genome in self.population.values():
                if best is None or genome.fitness > best.fitness:
                    best = genome

            fitnesses = [genome.fitness for genome in self.population.values()]
            fitness_mean = mean(fitnesses)
            fitness_stdev = stdev(fitnesses)
            best_species_id = self.species.get_species_id(best.genome_id)
            print(f'POPULATIONS AVERAGE FITNESS: {fitness_mean:3.5f}, STDEV: {fitness_stdev:3.5f}')
            print(f'BEST FITNESS: {best.fitness:3.5f}, SIZE: {best.size()}, SPECIES: {best_species_id}, '
                  f'ID: {best.genome_id}')

            if self.best is None or best.fitness > self.best.fitness:
                self.best = best

            # end if the fitness threshold is reached (if available)
            if self.config.fitness_threshold is not None:
                if self.best.fitness >= self.config.fitness_threshold:
                    print(f'\nBEST INDIVIDUAL IN GENERATION {self.generation}, COMPLEXITY: {self.best.size()}')
                    print('++++++++++ PROBLEM SOLVED! ++++++++++')
                    break

            # create new generation from current
            self.population = self.reproduction.reproduce(self.config, self.species, self.config.population_size,
                                                          self.generation)

            # check for dead population
            if not self.species.species:
                print('++++++++++ ALL SPECIES EXTINCT! ++++++++++')
                raise RuntimeError('Complete extinction exception!')

            # divide the new population into species
            self.species.speciate(self.config, self.population, self.generation)

            num_genomes = len(self.population)
            num_species = len(self.species.species)

            print(f'Population members: {num_genomes}\tNumber of species: {num_species}')
            species_ids = list(self.species.species.keys())
            species_ids.sort()
            print(f'\t ID \t AGE \t SIZE \t FITNESS \t ADJ FIT \t STAGNATION \n'
                  f'\t____\t ___ \t ____ \t _______ \t _______ \t __________ ')
            for species_id in species_ids:
                species = self.species.species[species_id]
                age = self.generation - species.created_generation
                size = len(species.members)
                fitness = 'NONE' if species.fitness is None else species.fitness
                adjusted_fitness = 'NONE' if species.adjusted_fitness is None else species.adjusted_fitness
                stagnation = self.generation - species.last_improved
                print(f'\t{species_id: >4}\t {age: >3} \t {size: >4} \t '
                      f'{fitness:.5f} \t {adjusted_fitness:.5f} \t {stagnation: >10}')

            time_spent = time.time() - self.generation_start
            self.generation_times.append(time_spent)
            self.generation_times = self.generation_times[-10:]
            average = sum(self.generation_times) / len(self.generation_times)
            if len(self.generation_times) > 1:
                print(f'GENERATION TIME: {time_spent:.3f}\tAVERAGE: {average:.3f}')
            else:
                print(f'GENERATION TIME: {time_spent:.3f}')

            self.generation += 1

        return self.best
