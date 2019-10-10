import random

from NEAT.src.connection_gene import ConnectionGene
from NEAT.src.node_gene import NodeGene


class Genome:
    def __init__(self):
        self.dict_of_connections = {}
        self.dict_of_nodes = {}

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
                                                0)  # No idea what is innovation number
                self.dict_of_connections[new_connection.innovation_num] = new_connection
            else:
                new_connection = ConnectionGene(node2.node_id, node1.node_id, random.random(), True,
                                                0)  # No idea what is innovation number
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
        in_node = connection_gene.in_node
        out_node = connection_gene.out_node

        connection_gene.disable()

        new_node = NodeGene('HIDDEN', len(self.dict_of_nodes))

        # Connect existing nodes with new one
        connection_in_to_new = ConnectionGene(in_node.node_id, new_node.node_id, 1.0, True, 0)  # innovation number
        connection_new_to_out = ConnectionGene(new_node.node_id, out_node.node_id, connection_gene.weight, True,
                                               0)  # innovation number

        self.dict_of_nodes[new_node.node_id] = new_node
        self.dict_of_connections[connection_in_to_new.innovation_num] = connection_in_to_new
        self.dict_of_connections[connection_new_to_out.innovation_num] = connection_new_to_out

    # parent_genome1 is more fit of parent
    def crossover(self, parent_genome1, parent_genome2):
        child_genome = Genome()

        for node in parent_genome1.dict_of_nodes.values():
            child_genome.dict_of_nodes[node.node_id] = node.copy()

        for connection in parent_genome1.dict_of_connections.values():
            if connection.innovation_num in parent_genome2.dict_of_connections:  # matching connection
                child_connection = connection.copy() if bool(random.getrandbits(1)) else \
                    parent_genome2.dict_of_connections[connection.innovation_num].copy()
                child_genome.dict_of_connections[0] = child_connection  # innovation number
            else:  # disjoint or exceed connection
                child_genome.dict_of_connections[0] = connection.copy()  # innovation number

        return child_genome
