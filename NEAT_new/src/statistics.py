from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


class Statistics:
    def __init__(self):
        self.most_fit_genomes = []
        self.generation_statistics = []

    def print_stats(self, filename='population_stats.svg'):
        generations = range(len(self.most_fit_genomes))
        best_fitnesses = [genome.fitness for genome in self.most_fit_genomes]
        average_fitnesses = np.array(self.get_fitness_stat(mean))
        stdev_fitnesses = np.array(self.get_fitness_stat(stdev))

        plt.plot(generations, average_fitnesses, 'b-', label='AVERAGE')
        plt.plot(generations, average_fitnesses - stdev_fitnesses, 'g-', label='-1 STDEV')
        plt.plot(generations, average_fitnesses + stdev_fitnesses, 'g-', label='+1 STDEV')
        plt.plot(generations, best_fitnesses, 'r-', label='BEST')

        plt.title('POPULATION STATISTICS')
        plt.xlabel('GENERATIONS')
        plt.ylabel('FITNESS')
        plt.grid()
        plt.legend()
        plt.savefig(filename)
        plt.show()
        plt.close()

    def draw_trains(self):
        pass

    def draw_species(self, filename='species.svg'):
        species_size = self.get_species_sizes()
        generations = len(species_size)
        curves = np.array(species_size).T

        figure, ax = plt.subplots()
        ax.stackplot(range(generations), *curves)

        plt.title('SPECIES')
        plt.ylabel('SIZE OF SPECIES')
        plt.xlabel('GENERATIONS')
        plt.savefig(filename)
        plt.show()
        plt.close()

    def draw_genome(self):
        pass

    def get_fitness_stat(self, function):
        stat = []
        for stats in self.generation_statistics:
            scores = []
            for species_stats in stats.values():
                scores.extend(species_stats.values())
            stat.append(function(scores))
        return stat

    def get_species_sizes(self):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_counts = []
        for gen_data in self.generation_statistics:
            species = [len(gen_data.get(sid, [])) for sid in range(1, max_species + 1)]
            species_counts.append(species)

        return species_counts
