import random
from NEAT.src.connection_gene import ConnectionGene
from NEAT.src.node_gene import NodeGene


class Genome:
    def __init__(self):
        self.list_of_connections = []
        self.list_of_nodes = []

    # Structural Mutations
    def add_connection_mutation(self):
        randoms = random.sample(range(0, len(self.list_of_nodes)), 2)
        node1 = self.list_of_nodes[randoms[0]]
        node2 = self.list_of_nodes[randoms[1]]

        in_correct_order = self.are_nodes_in_correct_order(node1, node2)
        connection_exist = self.is_connection(node1, node2)

        if not connection_exist:
            if in_correct_order:
                new_connection = ConnectionGene(node1.node_id, node2.node_id, random.random(), True,
                                                0)  # No idea what is innovation number
                self.list_of_connections.append(new_connection)
            else:
                new_connection = ConnectionGene(node2.node_id, node1.node_id, random.random(), True,
                                                0)  # No idea what is innovation number
                self.list_of_connections.append(new_connection)

    def is_connection(self, node1, node2):
        connection_exist = False
        for connection in self.list_of_connections:
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
        connection_gene = self.list_of_connections[random.randint(0, len(self.list_of_connections) - 1)]
        in_node = connection_gene.in_node
        out_node = connection_gene.out_node

        connection_gene.disable()

        new_node = NodeGene('HIDDEN', len(self.list_of_nodes))

        # Connect existing nodes with new one
        connection_in_to_new = ConnectionGene(in_node.node_id, new_node.node_id, 1.0, True, 0)  # innovation number
        connection_new_to_out = ConnectionGene(new_node.node_id, out_node.node_id, connection_gene.weight, True,
                                               0)  # innovation number

        self.list_of_nodes.append(new_node)
        self.list_of_connections.append(connection_in_to_new)
        self.list_of_connections.append(connection_new_to_out)