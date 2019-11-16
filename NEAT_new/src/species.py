from itertools import count
from statistics import mean, stdev


class Species:
    def __init__(self, species_id, generation):
        self.species_id = species_id
        self.members = {}
        self.created_generation = generation
        self.last_improved = generation
        self.mascot = None
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.fitness_history = []

    def updated_species(self, mascot, members):
        self.mascot = mascot
        self.members = members

    def get_members_fitness(self):
        members_fitness = [member.fitness for member in self.members.values()]
        return members_fitness


class SpeciesSet:
    def __init__(self):
        self.species = {}
        self.indexer = count(1)
        self.genome_to_species = {}

    def speciate(self, config, population, generation):
        assert isinstance(population, dict)

        compatibility_threshold = config.compatibility_threshold

        # find species leaders
        not_in_species = set(population.keys())
        distances = Distances(config)
        new_mascots = {}
        new_members = {}

        for species_id, species in self.species.items():
            candidates = []
            for genome_id in not_in_species:
                genome = population[genome_id]
                distance = distances(species.mascot, genome)
                candidates.append((distance, genome))

            # find closest genome to current mascot
            _, mascot = min(candidates, key=lambda x: x[0])
            mascot_id = mascot.genome_id
            new_mascots[species_id] = mascot_id
            new_members[species_id] = [mascot_id]
            not_in_species.remove(mascot_id)

        # divide whole population into species based on distance
        while not_in_species:
            genome_id = not_in_species.pop()
            genome = population[genome_id]

            # find species with closest mascot
            candidates = []
            for species_id, mascot_id in new_mascots.items():
                mascot = population[mascot_id]
                distance = distances(mascot, genome)
                if distance < compatibility_threshold:
                    candidates.append((distance, species_id))

            if candidates:
                _, species_id = min(candidates, key=lambda x: x[0])
                new_members[species_id].append(genome_id)
            else:
                # use this genome as mascot
                species_id = next(self.indexer)
                new_mascots[species_id] = genome_id
                new_members[species_id] = [genome_id]

        # update species collection with new one
        self.genome_to_species = {}
        for species_id, mascot_id in new_mascots.items():
            species = self.species.get(species_id)
            if species is None:
                species = Species(species_id, generation)
                self.species[species_id] = species

            members = new_members[species_id]
            for genome_id in members:
                self.genome_to_species[genome_id] = species_id

            members_dict = {genome_id: population[genome_id] for genome_id in members}
            species.updated_species(population[mascot_id], members_dict)

        genomes_distance_mean = mean(distances.distances.values())  # not sure it will work
        genomes_distance_stdev = stdev(distances.distances.values())  # same

        # report!

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]


class Distances:
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome1, genome2):
        g1_id = genome1.genome_id
        g2_id = genome2.genome_id
        distance = self.distances.get((g1_id, g2_id))
        if distance is None:
            distance = genome1.distance(genome2, self.config)
            self.distances[g1_id, g2_id] = distance
            self.distances[g2_id, g1_id] = distance
            self.misses += 1
        else:
            self.hits += 1

        return distance
