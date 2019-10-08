import unittest
from parameterized import parameterized
from NEAT.src.node_gene import NodeGene
from NEAT.src.connection_gene import ConnectionGene
from NEAT.src.genome import Genome


class GenomeTestCase(unittest.TestCase):
    @parameterized.expand([
        [NodeGene('SENSOR', 1), NodeGene('SENSOR', 2), True],
        [NodeGene('SENSOR', 1), NodeGene('HIDDEN', 2), True],
        [NodeGene('SENSOR', 1), NodeGene('OUTPUT', 2), True],
        [NodeGene('HIDDEN', 1), NodeGene('SENSOR', 2), False],
        [NodeGene('HIDDEN', 1), NodeGene('HIDDEN', 2), True],
        [NodeGene('HIDDEN', 1), NodeGene('OUTPUT', 2), True],
        [NodeGene('OUTPUT', 1), NodeGene('SENSOR', 2), False],
        [NodeGene('OUTPUT', 1), NodeGene('HIDDEN', 2), False],
        [NodeGene('OUTPUT', 1), NodeGene('OUTPUT', 2), True],
    ])
    def test_are_nodes_in_correct_order(self, node1, node2, output):
        genome = Genome()
        in_order = genome.are_nodes_in_correct_order(node1, node2)
        self.assertEqual(in_order, output)

    @parameterized.expand([
        [NodeGene('SENSOR', 1), NodeGene('SENSOR', 2), False],
        [NodeGene('SENSOR', 2), NodeGene('HIDDEN', 3), True],
        [NodeGene('SENSOR', 3), NodeGene('HIDDEN', 2), True],
    ])
    def test_is_connection(self, node1, node2, output):
        genome = Genome()
        genome.list_of_connections.append(ConnectionGene(1, 3, 0.5, True, 1))
        genome.list_of_connections.append(ConnectionGene(2, 3, 0.5, True, 2))
        connection_exists = genome.is_connection(node1, node2)

        self.assertEqual(connection_exists, output)


if __name__ == '__main__':
    unittest.main()
