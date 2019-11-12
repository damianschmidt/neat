class Gene:
    def __init__(self, gene_id):
        self.gene_id = gene_id


class NodeGene(Gene):
    def distance(self, other_node):
        pass


class ConnectionGene(Gene):
    def distance(self, other):
        pass
