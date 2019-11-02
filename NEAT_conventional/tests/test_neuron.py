import unittest

from parameterized import parameterized

from NEAT_conventional.src.neuron import Neuron


class NeuronTestCase(unittest.TestCase):
    def test_add_output_connections(self):
        neuron = Neuron()
        neuron.add_output_connections(1, 0.5)
        self.assertEqual(neuron.output_ids[0], 1)
        self.assertEqual(neuron.output_weights[0], 0.5)

    def test_add_input_connections(self):
        neuron = Neuron()
        neuron.add_input_connections()
        self.assertEqual(len(neuron.inputs), 1)
        self.assertEqual(neuron.inputs[0], None)

    def test_calculate(self):
        neuron = Neuron()
        neuron.inputs = [0.2, 0.3]
        result = neuron.calculate()
        self.assertEqual(result, 0.9205614508160216)

    @parameterized.expand([[[1, 2, 3], True], [[1, None, 2], False]])
    def test_is_ready_to_calculate(self, inputs, output):
        neuron = Neuron()
        neuron.inputs = inputs
        result = neuron.is_ready_to_calculate()
        self.assertEqual(result, output)

    def test_feed_input(self):
        neuron = Neuron()
        neuron.inputs = [1, None, 2]
        neuron.feed_input(3)
        self.assertEqual(neuron.inputs, [1, 3, 2])

    def test_feed_input_no_slot(self):
        neuron = Neuron()
        neuron.inputs = [1, 2]
        with self.assertRaises(RuntimeError):
            neuron.feed_input(3)

    def test_reset(self):
        neuron = Neuron()
        neuron.inputs = [1, 2, 3]
        neuron.output = 0.71
        neuron.reset()
        self.assertEqual(neuron.inputs, [None, None, None])
        self.assertEqual(neuron.output, 0.0)

    @parameterized.expand([[0.0, 0.5], [0.5, 0.9205614508160216]])
    def test_sigmoid_activation_function(self, total_sum, output):
        neuron = Neuron()
        result = neuron.sigmoid_activation_function(total_sum)
        self.assertEqual(result, output)
