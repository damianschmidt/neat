from NEAT_new.src.connection import ConnectionPheno
from NEAT_new.src.node import NodePheno


class Network:
    def __init__(self, genome):
        self.genome = genome
        self.nodes = []
        self.connections = []

        for node_gene in genome.nodes:
            self.nodes.append(NodePheno(node_gene))

        for connection_gene in genome.connections:
            if not connection_gene.disabled:
                self.connections.append(ConnectionPheno(self.nodes, connection_gene))
