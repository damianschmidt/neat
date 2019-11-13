class Gene:
    def __init__(self, gene_id):
        self.gene_id = gene_id

    def crossover(self, gene):
        assert self.gene_id == gene.gene_id
        pass

    def copy(self):
        pass


class NodeGene(Gene):
    def __init__(self, gene_id):
        super().__init__(gene_id)
        self.bias = 0.0
        self.response = 0.0
        self.activation = 'sigmoid'
        self.aggregation = 'sum'

    def distance(self, other_node):
        pass


class ConnectionGene(Gene):
    def __init__(self, gene_id):
        super().__init__(gene_id)
        self.weight = 0.0
        self.enabled = True

    def distance(self, other):
        pass
