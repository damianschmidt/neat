import unittest
from unittest.mock import patch, MagicMock

from NEAT_conventional.src.evaluator import Evaluator
from NEAT_conventional.src.fitness_genome import FitnessGenome
from NEAT_conventional.src.genome import Genome
from NEAT_conventional.src.innovation_generator import InnovationGenerator
from NEAT_conventional.src.species import Species


class TestEvaluator(Evaluator):
    def evaluate_genome(self, genome):
        return 1


class EvaluatorTestCase(unittest.TestCase):
    def setUp(self):
        self.starting_genome = Genome()
        node_innovation = InnovationGenerator
        con_innovation = InnovationGenerator
        self.evaluator = TestEvaluator(self.starting_genome, node_innovation, con_innovation)

    def test_list_of_starting_genomes(self):
        self.evaluator.population_size = 5
        result = self.evaluator.list_of_starting_genomes(self.starting_genome)
        self.assertEqual(5, len(result))

    def test_reset_before_generation(self):
        self.evaluator.genome_fitness[0] = 1
        self.evaluator.genome_species[0] = 1
        self.evaluator.next_generation.append(1)
        self.evaluator.highest_score = 1
        self.evaluator.fittest_genome = Genome()
        self.evaluator.reset_before_generation()

        self.assertEqual({}, self.evaluator.genome_fitness)
        self.assertEqual({}, self.evaluator.genome_species)
        self.assertEqual([], self.evaluator.next_generation)
        self.assertEqual(0, self.evaluator.highest_score)
        self.assertEqual(None, self.evaluator.fittest_genome)

    @patch('NEAT_conventional.src.genome_utils.compatibility_distance', MagicMock(return_value=5.0))
    def test_place_genomes_into_species__found_species(self):
        species = Species(self.evaluator.genomes[0])
        self.evaluator.species.append(species)
        self.evaluator.place_genomes_into_species()
        self.assertEqual(1001, len(self.evaluator.species[0].members))
        self.assertEqual(1, len(self.evaluator.genome_species))

    @patch('NEAT_conventional.src.genome_utils.compatibility_distance', MagicMock(return_value=15.0))
    def test_place_genomes_into_species__not_found_species(self):
        species = Species(self.evaluator.genomes[0])
        self.evaluator.species.append(species)
        self.evaluator.place_genomes_into_species()
        self.assertEqual(1001, len(self.evaluator.species))
        self.assertEqual(1, len(self.evaluator.genome_species))

    def test_remove_species_without_genomes(self):
        species = Species(self.evaluator.genomes[0])
        species.members = []
        self.evaluator.species.append(species)
        self.evaluator.remove_species_without_genomes()
        self.assertEqual(0, len(self.evaluator.species))

    @patch('NEAT_conventional.src.genome_utils.compatibility_distance', MagicMock(return_value=5.0))
    def test_evaluate_genomes_and_assign_fitness(self):
        species = Species(self.evaluator.genomes[0])
        self.evaluator.species.append(species)
        self.evaluator.place_genomes_into_species()
        self.evaluator.evaluate_genomes_and_assign_fitness()

        self.assertEqual(1, self.evaluator.highest_score)
        self.assertEqual(self.evaluator.genomes[0], self.evaluator.fittest_genome)
        self.assertEqual(1, len(self.evaluator.genome_fitness))

    def test_best_into_next_generation(self):
        species1 = Species(self.evaluator.genomes[0])
        fit_pop1 = FitnessGenome(self.evaluator.genomes[0], 15.0)
        fit_pop2 = FitnessGenome(self.evaluator.genomes[0], 12.0)
        species1.fitness_population.append(fit_pop1)
        species1.fitness_population.append(fit_pop2)

        species2 = Species(self.evaluator.genomes[1])
        fit_pop3 = FitnessGenome(self.evaluator.genomes[0], 11.0)
        fit_pop4 = FitnessGenome(self.evaluator.genomes[0], 14.0)
        species2.fitness_population.append(fit_pop3)
        species2.fitness_population.append(fit_pop4)

        self.evaluator.species.append(species1)
        self.evaluator.species.append(species2)

        self.evaluator.best_into_next_generation()

        result = [fit_pop1.genome, fit_pop4.genome]
        self.assertEqual(result, self.evaluator.next_generation)

    # def test_remove_stagnation(self):
    #     pass

    # test for filling rest of next generation
    # def test_breed_rest_of_genomes
    # pass
