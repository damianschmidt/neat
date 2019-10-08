from random import sample


class Genome:
    def __init__(self):
        self.list_of_connections = []
        self.list_of_nodes = []

    # Structural Mutations
    def add_connection_mutation(self):
        randoms = sample(range(0, len(self.list_of_nodes) - 1), 2)
        node1 = self.list_of_nodes[randoms[0]]
        node2 = self.list_of_nodes[randoms[1]]

        in_correct_order = self.are_nodes_in_correct_order(node1, node2)
        connection_exist = self.is_connection(node1, node2)

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
        pass
