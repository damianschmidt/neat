from NEAT_new.src.report import ReporterSet
from NEAT_new.src.reproduction import Reproduction
from NEAT_new.src.species import SpeciesSet
from NEAT_new.src.stagnation import Stagnation


class GeneticAlgorithm:
    def __init__(self, config):
        self.generation = 0
        self.best = None
        self.config = config
        self.reporters = ReporterSet()
        self.stagnation = Stagnation(config)
        self.reproduction = Reproduction(self.stagnation, self.reporters)
        self.population = self.reproduction.create_new(config.population_size)
        self.species = SpeciesSet(self.reporters)
        self.species.speciate(config, self.population, self.generation)

    def run(self, fitness_function, epochs=None):
        i = 0
        while epochs is None or i < epochs:
            i += 1

            self.reporters.start_generation(self.generation)

            # evaluate all genomes
            fitness_function(list(self.population.items()), self.config)

            # get best and report
            best = None
            for genome in self.population.values():
                if best is None or genome.fitness > best.fitness:
                    best = genome
            self.reporters.post_evaluate(self.population, self.species, best)

            if self.best is None or best.fitness > self.best.fitness:
                self.best = best

            # end if the fitness threshold is reached (if available)
            if self.config.fitness_threshold is not None:
                if self.best.fitness >= self.config.fitness_threshold:
                    self.reporters.found_solution(best)
                    break

            # create new generation from current
            self.population = self.reproduction.reproduce(self.config, self.species, self.config.population_size,
                                                          self.generation)

            # check for dead species
            if not self.species.species:
                self.reporters.complete_extinction()
                raise RuntimeError('Complete extinction exception!')

            # divide the new population into species
            self.species.speciate(self.config, self.population, self.generation)
            self.reporters.end_generation(self.population, self.species)
            self.generation += 1

        return self.best
