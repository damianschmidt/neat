from itertools import count

from NEAT_new.src.genome import Genome


class Reproduction:
    def __init__(self):
        self.genome_indexer = count(1)
        self.ancestors = {}

    def create_new(self, num_genome):
        new_genomes = {}
        for i in range(num_genome):
            genome_id = next(self.genome_indexer)
            g = Genome(genome_id)
            new_genomes[genome_id] = g
            self.ancestors[genome_id] = tuple()

        return new_genomes

    def reproduce(self):
        pass
