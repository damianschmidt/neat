import unittest
from NEAT_legacy.src.neural_network import NeuralNetwork, Neuron
from NEAT_legacy.src.genome import Genome


class NeuralNetworkTestCase(unittest.TestCase):
    def setUp(self):
        genome = Genome()
        self.network = NeuralNetwork(genome)

    def test_calculate_wrong_input(self):
        self.network.input = [1, 2]
        with self.assertRaises(ValueError):
            self.network.calculate(input_parameters=[0.5, 0.7, 0.8])

    def test_reset_network(self):
        # There is no sense to testing neurons reset cause I have checked it in test_neuron
        neuron1 = Neuron()
        neuron2 = Neuron()
        self.network.neurons = {1: neuron1, 2: neuron2}
        self.network.unprocessed = [neuron1]
        self.network.reset_network()
        self.assertEqual(2, len(self.network.unprocessed))

    # def test_prepare_inputs(self):
    #     pass
    #
    # def test_solve_network(self):
    #     pass
