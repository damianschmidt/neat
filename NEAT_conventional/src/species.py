import random


class Species:
    def __init__(self, mascot):
        self.mascot = mascot
        self.members = [self.mascot]
        self.fitness_population = []
        self.total_adjusted_fitness = 0.0

    def reset(self):
        new_mascot_index = random.randint(0, len(self.members) - 1)
        self.mascot = self.members[new_mascot_index]
        self.members = [self.mascot]
        self.fitness_population = []
        self.total_adjusted_fitness = 0.0
