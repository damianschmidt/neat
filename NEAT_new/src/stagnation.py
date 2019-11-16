class Stagnation:
    def __init__(self, config):
        self.config = config
        self.species_fitness_func = config.species_fitness_func

    def update(self, species_set, generation):
        species_data = []
        for species_id, species in species_set.species.items():
            if species.fitness_history:
                previous_fitness = max(species.fitness_history)
            else:
                previous_fitness = 0.0

            species_fitnesses = [member.fitness for member in species.members.values()]
            species.fitness = self.species_fitness_func(species_fitnesses)
            species.fitness_history.append(species.fitness)
            species.adjusted_fitness = None
            if previous_fitness is None or species.fitness > previous_fitness:
                species.last_improved = generation

            species_data.append((species_id, species))

        # Sort
        species_data.sort(key=lambda x: x[1].fitness)

        result = []
        number_of_non_stag = len(species_data)
        for index, (species_id, species) in enumerate(species_data):
            time_of_stag = generation - species.last_improved
            is_stag = False
            if number_of_non_stag > self.config.elitism:
                is_stag = time_of_stag >= self.config.max_stagnation

            if (len(species_data) - index) <= self.config.elitism:
                is_stag = False

            if is_stag:
                number_of_non_stag -= 1

            result.append((species_id, species, is_stag))
        return result
