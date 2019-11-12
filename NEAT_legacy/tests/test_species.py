import unittest

from NEAT_legacy.src.fitness_genome import FitnessGenome
from NEAT_legacy.src.genome import Genome
from NEAT_legacy.src.species import Species


class SpeciesTestCase(unittest.TestCase):
    def test_reset(self):
        mascot1 = Genome()
        mascot2 = Genome()
        fitness_pop1 = FitnessGenome(mascot1, 0.15)
        fitness_pop2 = FitnessGenome(mascot2, 0.12)
        species = Species(mascot1)
        species.members.append(mascot2)
        species.fitness_population = [fitness_pop1, fitness_pop2]
        species.total_adjusted_fitness = 0.27
        species.reset()

        self.assertEqual(1, len(species.members))
        self.assertEqual(0, len(species.fitness_population))
        self.assertEqual(0.0, species.total_adjusted_fitness)
