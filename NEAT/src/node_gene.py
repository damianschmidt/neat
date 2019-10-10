class NodeGene:
    def __init__(self, node_type, node_id):
        self.node_id = node_id
        self.node_type = node_type

    def copy(self):
        return NodeGene(self.node_type, self.node_id)
