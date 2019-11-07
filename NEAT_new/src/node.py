class NodeGene:
    def __init__(self, node_id, node_type, activation_response=None):
        self.node_id = node_id
        self.node_type = node_type
        self.activation_response = 1 / 4.924273 if activation_response is None else activation_response  # curvature of sigmoid
        # self.innovation_num = innovation_num


class NodePheno:
    def __init__(self, node_gene):
        self.node_id = node_gene.node_id
        self.node_type = node_gene.node_type
        self.input_connections = []
        self.output_connections = []
        self.sum_activation = 0.0
        self.node_output = 0.0
        self.activation_response = node_gene.activation_response
