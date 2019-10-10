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
        [NodeGene('SENSOR', 0), NodeGene('SENSOR', 1), False],
        [NodeGene('SENSOR', 1), NodeGene('HIDDEN', 2), True],
        [NodeGene('SENSOR', 2), NodeGene('HIDDEN', 1), True],
    ])
    def test_is_connection(self, node1, node2, output):
        genome = Genome()
        genome.dict_of_connections[0] = ConnectionGene(0, 2, 0.5, True, 0)
        genome.dict_of_connections[1] = ConnectionGene(1, 2, 0.5, True, 1)
        connection_exists = genome.is_connection(node1, node2)

        self.assertEqual(connection_exists, output)

    def test_add_connection_mutation(self):
        genome = Genome()
        genome.dict_of_nodes[0] = NodeGene('SENSOR', 0)
        genome.dict_of_nodes[1] = NodeGene('OUTPUT', 1)

        genome.add_connection_mutation()

        self.assertEqual(len(genome.dict_of_connections), 1)

    def test_add_node_mutation(self):
        genome = Genome()

        genome.dict_of_connections[0] = ConnectionGene(0, 2, 0.5, True, 0)
        genome.dict_of_connections[1] = ConnectionGene(1, 2, 0.5, True, 1)

        genome.dict_of_nodes[0] = NodeGene('SENSOR', 0)
        genome.dict_of_nodes[1] = NodeGene('SENSOR', 1)
        genome.dict_of_nodes[2] = NodeGene('OUTPUT', 2)

        genome.add_node_mutation()

        self.assertEqual(len(genome.dict_of_nodes), 4)
        self.assertEqual(len(genome.dict_of_connections), 4)
        self.assertEqual(genome.dict_of_nodes[3].node_type, 'HIDDEN')

    def test_crossover_without_connections(self):
        """ Only with same parent genomes cause of problem with testing random choice of matching child connection"""
        parent_genome1 = Genome()
        parent_genome1.dict_of_nodes[0] = NodeGene('SENSOR', 0)
        parent_genome1.dict_of_nodes[1] = NodeGene('SENSOR', 1)
        parent_genome1.dict_of_nodes[2] = NodeGene('OUTPUT', 2)

        parent_genome2 = Genome()
        parent_genome2.dict_of_nodes[0] = NodeGene('SENSOR', 0)
        parent_genome2.dict_of_nodes[1] = NodeGene('SENSOR', 1)
        parent_genome2.dict_of_nodes[2] = NodeGene('OUTPUT', 2)

        genome = Genome()
        child_genome = genome.crossover(parent_genome1, parent_genome2)

        self.assertEqual(parent_genome1.dict_of_nodes, child_genome.dict_of_nodes)
        self.assertEqual(parent_genome1.dict_of_connections, child_genome.dict_of_connections)

    def test_crossover_disjoint_and_exceed(self):
        """ Only with same parent genomes cause of problem with testing random choice of matching child connection"""
        parent_genome1 = Genome()
        parent_genome1.dict_of_nodes[0] = NodeGene('SENSOR', 0)
        parent_genome1.dict_of_nodes[1] = NodeGene('SENSOR', 1)
        parent_genome1.dict_of_nodes[2] = NodeGene('OUTPUT', 2)
        parent_genome1.dict_of_nodes[3] = NodeGene('HIDDEN', 3)

        parent_genome1.dict_of_connections[0] = ConnectionGene(0, 2, 0.5, False, 0)

        parent_genome2 = Genome()
        parent_genome2.dict_of_nodes[0] = NodeGene('SENSOR', 0)
        parent_genome2.dict_of_nodes[1] = NodeGene('SENSOR', 1)
        parent_genome2.dict_of_nodes[2] = NodeGene('OUTPUT', 2)
        parent_genome1.dict_of_nodes[3] = NodeGene('HIDDEN', 3)

        genome = Genome()
        child_genome = genome.crossover(parent_genome1, parent_genome2)

        self.assertEqual(parent_genome1.dict_of_nodes, child_genome.dict_of_nodes)
        self.assertEqual(parent_genome1.dict_of_connections, child_genome.dict_of_connections)


if __name__ == '__main__':
    unittest.main()
