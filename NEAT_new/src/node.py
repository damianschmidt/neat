class NodeType:
    INPUT, OUTPUT, HIDDEN, BIAS = range(4)


class NodeGene:
    def __init__(self, node_id, node_type, pos_x, pos_y, activation_response=None):
        self.node_id = node_id
        self.node_type = node_type
        self.activation_response = 1/4.924273 if activation_response is None else activation_response

        # visualize
        self.pos_x = pos_x
        self.pos_y = pos_y

    def __repr__(self):
        return f'(ID:{self.node_id}, Type:{self.node_type})'


class NodeNet:
    def __init__(self, node_gene):
        self.node_id = node_gene.node_id
        self.node_type = node_gene.node_type
        self.input_connections = []
        self.output_connections = []
        self.sum_activation = 0.0
        self.value = 0.0
        self.activation_response = node_gene.activation_response

        # visualize
        self.pos_x = node_gene.pos_x
        self.pos_y = node_gene.pos_y
