import unittest

from NEAT.src.evaluator import Evaluator
from NEAT.src.genome import Genome


class SpeciesTestCase(unittest.TestCase):
    def test_reset(self):
        mascot1 = Genome()
        mascot2 = Genome()
        fitness_pop1 = Evaluator.FitnessGenome(mascot1, 0.15)
        fitness_pop2 = Evaluator.FitnessGenome(mascot2, 0.12)
        species = Evaluator.Species(mascot1)
        species.members.append(mascot2)
        species.fitness_population = [fitness_pop1, fitness_pop2]
        species.total_adjusted_fitness = 0.27
        species.reset()

        self.assertEqual(1, len(species.members))
        self.assertEqual(0, len(species.fitness_population))
        self.assertEqual(0.0, species.total_adjusted_fitness)
