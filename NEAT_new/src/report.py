import time
from statistics import mean, stdev


class ReporterSet:
    def __init__(self):
        self.reporters = []

    def add(self, reporter):
        self.reporters.append(reporter)

    def remove(self, reporter):
        self.reporters.remove(reporter)

    def start_generation(self, genome):
        for r in self.reporters:
            r.start_generation(genome)

    def end_generation(self, population, species_set):
        for r in self.reporters:
            r.end_generation(population, species_set)

    def post_evaluate(self, population, species, best_genome):
        for r in self.reporters:
            r.post_evaluate(population, species, best_genome)

    def complete_extinction(self):
        for r in self.reporters:
            r.complete_extinction()

    def found_solution(self, best):
        for r in self.reporters:
            r.found_solution(best)

    def species_stagnant(self, species_id, species):
        for r in self.reporters:
            r.species_stagnant(species_id, species)

    def info(self, msg):
        for r in self.reporters:
            r.info(msg)


class Reporter:
    def __init__(self, species_details):
        self.species_details = species_details
        self.generation = None
        self.generation_start = None
        self.generation_times = []
        self.number_of_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        print(f'\n ++++++++++ GENERATION {generation} ++++++++++ \n')
        self.generation_start = time.time()

    def end_generation(self, population, species_set):
        num_genomes = len(population)
        num_species = len(species_set.species)
        if self.species_details:
            print(f'Population members: {num_genomes}\tNumber of species: {num_species}')
            species_ids = list(species_set.species.keys())
            species_ids.sort()
            print(f'\t ID \t AGE \t SIZE \t FITNESS \t ADJ FIT \t STAGNATION \n'
                  f'\t____\t ___ \t ____ \t _______ \t _______ \t __________ ')
            for species_id in species_ids:
                species = species_set.species[species_id]
                age = self.generation - species.created_generation
                size = len(species.members)
                fitness = 'NONE' if species.fitness is None else species.fitness
                adjusted_fitness = 'NONE' if species.adjusted_fitness is None else species.adjusted_fitness
                stagnation = self.generation - species.last_improved
                print(
                    f'\t{species_id: >4}\t {age: >3} \t {size: >4} \t {fitness:.5f} \t {adjusted_fitness:.5f} \t {stagnation: >10}')
        else:
            print(f'Population members: {num_genomes}\tNumber of species: {num_species}')

        time_spent = time.time() - self.generation_start
        self.generation_times.append(time_spent)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print(f'TOTAL EXTINCTIONS: {self.number_of_extinctions}')
        if len(self.generation_times) > 1:
            print(f'GENERATION TIME: {time_spent:.3f}\tAVERAGE: {average:.3f}')
        else:
            print(f'GENERATION TIME: {time_spent:.3f}')

    @staticmethod
    def post_evaluate(population, species, best_genome):
        fitnesses = [genome.fitness for genome in population.values()]
        fitness_mean = mean(fitnesses)
        fitness_stdev = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.genome_id)
        print(f'POPULATIONS AVERAGE FITNESS: {fitness_mean:3.5f}, STDEV: {fitness_stdev:3.5f}')
        print(
            f'BEST FITNESS: {best_genome.fitness:3.5f}, SIZE: {best_genome.size()}, SPECIES: {best_species_id}, ID: {best_genome.genome_id}')

    def complete_extinction(self):
        self.number_of_extinctions += 1
        print('All species extinct.')

    def found_solution(self, best):
        print(f'\nBEST INDIVIDUAL IN GENERATION {self.generation}, COMPLEXITY: {best.size()}')

    def species_stagnant(self, species_id, species):
        if self.species_details:
            print(f'\nSPECIES {species_id} WITH {len(species.members)} MEMBERS IS STAGNANT: REMOVING IT')

    @staticmethod
    def info(msg):
        print(msg)
