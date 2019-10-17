import random

from NEAT.src.connection_gene import ConnectionGene
from NEAT.src.node_gene import NodeGene
from NEAT.src.innovation_generator import InnovationGenerator
from NEAT.src.neat_conf import PROBABILITY_OF_PERTURBING


class Genome:
    def __init__(self):
        self.dict_of_connections = {}
        self.dict_of_nodes = {}
        self.con_innovation = InnovationGenerator()
        self.node_innovation = InnovationGenerator()

    def mutation(self):
        for connection in self.dict_of_connections.values():
            if random.random() < PROBABILITY_OF_PERTURBING:
                connection.weight = connection.weight * random.uniform(-1, 1)
            else:
                connection.weight = random.uniform(-1, 1)

    # Structural Mutations
    def add_connection_mutation(self):
        randoms = random.sample(range(0, len(self.dict_of_nodes)), 2)
        node1 = self.dict_of_nodes[randoms[0]]
        node2 = self.dict_of_nodes[randoms[1]]

        in_correct_order = self.are_nodes_in_correct_order(node1, node2)
        connection_exist = self.is_connection(node1, node2)

        if not connection_exist:
            if in_correct_order:
                new_connection = ConnectionGene(node1.node_id, node2.node_id, random.random(), True,
                                                self.con_innovation.get_innovation())  # No idea what is innovation number
                self.dict_of_connections[new_connection.innovation_num] = new_connection
            else:
                new_connection = ConnectionGene(node2.node_id, node1.node_id, random.random(), True,
                                                self.con_innovation.get_innovation())
                self.dict_of_connections[new_connection.innovation_num] = new_connection

    def is_connection(self, node1, node2):
        connection_exist = False
        for connection in self.dict_of_connections.values():
            if connection.in_node == node1.node_id and connection.out_node == node2.node_id:  # existing connection
                connection_exist = True
                break
            elif connection.in_node == node2.node_id and connection.out_node == node1.node_id:  # existing connection
                connection_exist = True
                break

        return connection_exist

    def are_nodes_in_correct_order(self, node1, node2):
        in_order = True
        if node1.node_type == 'HIDDEN' and node2.node_type == 'SENSOR':
            in_order = False
        elif node1.node_type == 'OUTPUT' and node2.node_type == 'HIDDEN':
            in_order = False
        elif node1.node_type == 'OUTPUT' and node2.node_type == 'SENSOR':
            in_order = False
        return in_order

    def add_node_mutation(self):
        connection_gene = self.dict_of_connections[random.randint(0, len(self.dict_of_connections) - 1)]
        in_node = self.dict_of_nodes[connection_gene.in_node]
        out_node = self.dict_of_nodes[connection_gene.out_node]

        connection_gene.disable_connection()

        new_node = NodeGene('HIDDEN', self.node_innovation.get_innovation())
        self.dict_of_nodes[new_node.node_id] = new_node

        # Connect existing nodes with new one
        connection_in_to_new = ConnectionGene(in_node.node_id, new_node.node_id, 1.0, True,
                                              self.con_innovation.get_innovation())
        self.dict_of_connections[connection_in_to_new.innovation_num] = connection_in_to_new

        connection_new_to_out = ConnectionGene(new_node.node_id, out_node.node_id, connection_gene.weight, True,
                                               self.con_innovation.get_innovation())
        self.dict_of_connections[connection_new_to_out.innovation_num] = connection_new_to_out

    # parent_genome1 is more fit of parent
    def crossover(self, parent_genome1, parent_genome2):
        child_genome = Genome()

        for node in parent_genome1.dict_of_nodes.values():
            child_genome.dict_of_nodes[node.node_id] = node

        for connection in parent_genome1.dict_of_connections.values():
            if connection.innovation_num in parent_genome2.dict_of_connections:  # matching connection
                child_connection = connection if bool(random.getrandbits(1)) else \
                    parent_genome2.dict_of_connections[connection.innovation_num]
                child_genome.dict_of_connections[len(child_genome.dict_of_connections)] = child_connection
            else:  # disjoint or exceed connection
                child_genome.dict_of_connections[len(child_genome.dict_of_connections)] = connection

        return child_genome
