import math
import random
from itertools import count
from statistics import mean

from NEAT_new.src.genome import Genome


class Reproduction:
    def __init__(self, stagnation, config):
        self.genome_indexer = count(1)
        self.ancestors = {}
        self.stagnation = stagnation
        self.config = config

    def create_new(self, num_genome):
        new_genomes = {}
        for i in range(num_genome):
            genome_id = next(self.genome_indexer)
            g = Genome(genome_id)
            g.create_new(self.config)
            new_genomes[genome_id] = g
            self.ancestors[genome_id] = tuple()

        return new_genomes

    @staticmethod
    def spawn(adjusted_fitness, old_size, population_size, min_species_size):
        af_sum = sum(adjusted_fitness)
        spawn = []
        for af, os in zip(adjusted_fitness, old_size):
            if af_sum > 0:
                s = max(min_species_size, af / af_sum * population_size)
            else:
                s = min_species_size

            d = (s - os) * 0.5
            c = int(round(d))
            sp = os
            if abs(c) > 0:
                sp += c
            elif d > 0:
                sp += 1
            elif d < 0:
                sp -= 1

            spawn.append(sp)

        total_spawn = sum(spawn)
        normalize = population_size / total_spawn
        spawn = [max(min_species_size, int(round((n * normalize)))) for n in spawn]

        return spawn

    def reproduce(self, config, species, population_size, generation):
        all_fitnesses = []
        remaining_species = []
        for stag_species_id, stag_species, stag in self.stagnation.update(species, generation):
            if stag:
                print(f'\nSPECIES {stag_species_id} WITH {len(stag_species.members)} MEMBERS IS STAGNANT: REMOVING IT')
            else:
                all_fitnesses.extend(member.fitness for member in stag_species.members.values())
                remaining_species.append(stag_species)

        if not remaining_species:
            species.species = {}
            return {}

        min_fitness = min(all_fitnesses)
        max_fitness = max(all_fitnesses)

        fitness_range = max(1.0, max_fitness - min_fitness)
        for afs in remaining_species:  # afs - adjusted fitness species
            msf = mean([member.fitness for member in afs.members.values()])
            af = (msf - min_fitness) / fitness_range
            afs.adjusted_fitness = af

        adjusted_fitness = [species.adjusted_fitness for species in remaining_species]
        average_adjusted_fitness = mean(adjusted_fitness)
        print(f'AVERAGE ADJUSTED FITNESS: {average_adjusted_fitness:.3f}')

        # size of members for each species in the new generation
        old_size = [len(species.members) for species in remaining_species]
        min_species_size = config.elitism
        spawn_amounts = self.spawn(adjusted_fitness, old_size, population_size, min_species_size)

        new_population = {}
        species.species = {}

        for spawn, s in zip(spawn_amounts, remaining_species):
            spawn = max(spawn, config.elitism)

            old_members = list(s.members.items())
            s.members = {}
            species.species[s.species_id] = s

            # sort
            old_members.sort(reverse=True, key=lambda x: x[1].fitness)

            # transfer elites to new generation
            if config.elitism > 0:
                for i, m in old_members[:config.elitism]:
                    new_population[i] = m
                    spawn -= 1

            if spawn <= 0:
                continue

            # only use the survival threshold fraction to use as parents for the next generation
            reproduction_cutoff = int(math.ceil(config.survival_threshold * len(old_members)))

            # use at least two parents no matter what the threshold fraction result is
            reproduction_cutoff = max(reproduction_cutoff, 2)
            old_members = old_members[:reproduction_cutoff]

            # randomly choose parents and produce the number of offspring allotted to the species
            while spawn > 0:
                spawn -= 1

                parent1_id, parent1 = random.choice(old_members)
                parent2_id, parent2 = random.choice(old_members)

                genome_id = next(self.genome_indexer)
                child = Genome(genome_id)
                child.crossover(parent1, parent2)
                child.mutate(config)
                new_population[genome_id] = child
                self.ancestors[genome_id] = (parent1_id, parent2_id)

        return new_population
