class ConnectionGene:
    def __init__(self, in_node, out_node, weight, expressed, innovation_number):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight
        self.expressed = expressed
        self.innovation_num = innovation_number

    def disable_connection(self):
        self.expressed = False

    def copy(self):
        return ConnectionGene(self.in_node, self.out_node, self.weight, self.expressed, self.innovation_num)
