from random import random


class ConnectionGene:
    def __init__(self, in_node, out_node, innovation_num, disabled=False, weight=None, recurrent=False):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = random() * 2 - 1 if weight is None else weight
        self.disabled = disabled
        self.recurrent = recurrent  # TODO: necessary?
        self.innovation_num = innovation_num


class ConnectionNet:
    def __init__(self, nodes, connection_gene):
        self.input_node = next(filter(lambda n: n.node_id == connection_gene.in_node, nodes))
        self.output_node = next(filter(lambda n: n.node_id == connection_gene.out_node, nodes))
        self.input_node.output_connections.append(self)
        self.output_node.input_connections.append(self)
        self.weight = connection_gene.weight
