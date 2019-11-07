import numpy as np

from NEAT_new.src.connection import ConnectionNet
from NEAT_new.src.node import NodeNet
from NEAT_new.src.node_type import NodeType


class Network:
    def __init__(self, genome):
        self.genome = genome
        self.nodes = []
        self.connections = []

        for node_gene in genome.nodes:
            self.nodes.append(NodeNet(node_gene))

        for connection_gene in genome.connections:
            if not connection_gene.disabled:
                self.connections.append(ConnectionNet(self.nodes, connection_gene))

    def feed(self, inputs):
        outputs = []
        i_input = 0
        i_bias = 0
        for node in self.nodes:
            if node.node_type == NodeType.INPUT:
                node.value = inputs[i_input]
                i_input += 1
            elif node.node_type == NodeType.BIAS:
                node.value = 1
                i_bias += 1
            else:
                sum_w = 0.0
                for connection in node.input_connections:
                    sum_w += connection.weight * connection.input_node.value
                value = self.sigmoid(sum)
                node.value = value
                if node.node_type == NodeType.OUTPUT:
                    outputs.append(value)

        return np.array(outputs)

    @staticmethod
    def sigmoid(x):
        # -4.9 from documentation
        # *2 - 1 to have x(-1) = -1 and x(1) = 1
        return (2.0 / (1.0 + np.exp(-4.9 * x))) - 1.0
