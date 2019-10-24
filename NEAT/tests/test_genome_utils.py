import unittest
from unittest.mock import patch, MagicMock

from NEAT.src import genome_utils
from NEAT.src.genome import Genome
from NEAT.src.node_gene import NodeGene
from NEAT.src.connection_gene import ConnectionGene


class GenomeUtilsTestCase(unittest.TestCase):
    def setUp(self):
        self.genome1 = Genome()
        self.genome1.dict_of_nodes[1] = NodeGene('SENSOR', 1)
        self.genome1.dict_of_nodes[2] = NodeGene('SENSOR', 2)
        self.genome1.dict_of_nodes[3] = NodeGene('SENSOR', 3)
        self.genome1.dict_of_nodes[4] = NodeGene('OUTPUT', 4)
        self.genome1.dict_of_nodes[5] = NodeGene('HIDDEN', 5)
        self.genome1.dict_of_connections[1] = ConnectionGene(1, 4, 0.5, True, 1)
        self.genome1.dict_of_connections[2] = ConnectionGene(2, 4, 0.5, False, 2)
        self.genome1.dict_of_connections[3] = ConnectionGene(3, 4, 0.5, True, 3)
        self.genome1.dict_of_connections[4] = ConnectionGene(2, 5, 0.5, True, 4)
        self.genome1.dict_of_connections[5] = ConnectionGene(5, 4, 0.5, True, 5)
        self.genome1.dict_of_connections[8] = ConnectionGene(1, 5, 0.5, True, 8)

        self.genome2 = Genome()
        self.genome2.dict_of_nodes[1] = NodeGene('SENSOR', 1)
        self.genome2.dict_of_nodes[2] = NodeGene('SENSOR', 2)
        self.genome2.dict_of_nodes[3] = NodeGene('SENSOR', 3)
        self.genome2.dict_of_nodes[4] = NodeGene('OUTPUT', 4)
        self.genome2.dict_of_nodes[5] = NodeGene('HIDDEN', 5)
        self.genome2.dict_of_nodes[6] = NodeGene('HIDDEN', 6)
        self.genome2.dict_of_connections[1] = ConnectionGene(1, 4, 0.4, True, 1)
        self.genome2.dict_of_connections[2] = ConnectionGene(2, 4, 0.4, False, 2)
        self.genome2.dict_of_connections[3] = ConnectionGene(3, 4, 0.4, True, 3)
        self.genome2.dict_of_connections[4] = ConnectionGene(2, 5, 0.4, True, 4)
        self.genome2.dict_of_connections[5] = ConnectionGene(5, 4, 0.4, False, 5)
        self.genome2.dict_of_connections[6] = ConnectionGene(5, 6, 0.4, True, 6)
        self.genome2.dict_of_connections[7] = ConnectionGene(6, 4, 0.4, True, 7)
        self.genome2.dict_of_connections[9] = ConnectionGene(3, 5, 0.4, True, 9)
        self.genome2.dict_of_connections[10] = ConnectionGene(1, 6, 0.4, True, 10)

    @patch('NEAT.src.genome_utils.count_excess_genes', MagicMock(return_value=1))
    @patch('NEAT.src.genome_utils.count_disjoint_genes', MagicMock(return_value=2))
    @patch('NEAT.src.genome_utils.average_weight_diff', MagicMock(return_value=3.0))
    def test_compatibility_distance(self):
        result = genome_utils.compatibility_distance(self.genome1, self.genome2, 1.0, 2.0, 3.0)
        self.assertEqual(14.0, result)

    def test_count_matching_genes(self):
        result = genome_utils.count_matching_genes(self.genome1, self.genome2)
        self.assertEqual(5, result)

    def test_count_disjoint_genes(self):
        result = genome_utils.count_disjoint_genes(self.genome1, self.genome2)
        self.assertEqual(3, result)

    def test_count_excess_genes(self):
        result = genome_utils.count_excess_genes(self.genome1, self.genome2)
        self.assertEqual(3, result)

    def test_average_weight_diff(self):
        result = genome_utils.average_weight_diff(self.genome1, self.genome2)
        self.assertEqual(0.1, result)
