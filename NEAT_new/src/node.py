class NodeGene:
    def __init__(self, node_id, node_type):
        self.node_id = node_id
        self.node_type = node_type


class NodeNet:
    def __init__(self, node_gene):
        self.node_id = node_gene.node_id
        self.node_type = node_gene.node_type
        self.input_connections = []
        self.output_connections = []
        self.sum_activation = 0.0
        self.node_output = 0.0
