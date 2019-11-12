class GeneticAlgorithm:
    def __init__(self, initial_status):
        if initial_status is None:
            # self.population
            # self.species
            self.generation = 0
        else:
            self.population, self.species, self.generation = initial_status

        self.best = None

    def run(self, fitness_function, epochs=None):
        pass
