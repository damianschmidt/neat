from NEAT_new.src.reproduction import Reproduction
from NEAT_new.src.species import SpeciesSet
from NEAT_new.src.stagnation import Stagnation


class GeneticAlgorithm:
    def __init__(self, config):
        self.generation = 0
        self.best = None
        self.config = config
        self.stagnation = Stagnation(config)
        self.reproduction = Reproduction(self.stagnation)
        self.population = self.reproduction.create_new(config.population_size)
        self.species = SpeciesSet()
        self.species.speciate(config, self.population, self.generation)

    def run(self, fitness_function, epochs=None):
        i = 0
        while epochs is None or i < epochs:

            # report later

            # evaluate all genomes
            fitness_function(list(self.population.items()), self.config)

            # get best and report
            best = None
            for genome in self.population.values():
                if best is None or genome.fitness > best.fitness:
                    best = genome
            # report

            if self.best is None or best.fitness > self.best.fitness:
                self.best = best

            # end if the fitness threshold is reached (if available)
            if self.config.fitness_threshold is not None:
                if self.best.fitness >= self.config.fitness_threshold:
                    # FOUND SOLUTION REPORT
                    print('Fitness threshold achieved!')
                    break

            # create new generation from current
            self.population = self.reproduction.reproduce(self.config, self.species, self.config.population_size,
                                                          self.generation)

            # check for dead species
            # do it later

            # divide the new population into species
            self.species.speciate(self.config, self.population, self.generation)
            # report
            self.generation += 1

        return self.best
