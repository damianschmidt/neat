class InnovationType:
    NODE, CONNECTION = range(2)


class Innovation:
    def __init__(self, innovation_num, innovation_type, in_node=None, out_node=None, node_id=None, connection_id=None):
        self.innovation_num = innovation_num
        self.innovation_type = innovation_type
        self.in_node = in_node
        self.out_node = out_node
        self.node_id = node_id
        self.connection_id = connection_id


class InnovationSet:
    def __init__(self):
        self.innovations = []
        self.next_innovation_num = 0
        self.next_node_id = 0
        self.next_connection_id = 0

    def exist_innovation(self, innovation_type, in_node=None, out_node=None):
        for innovation in self.innovations:
            if innovation.innovation_type == innovation_type and innovation.in_node == in_node and innovation.out_node == out_node:
                return innovation
        return None

    def create_innovation(self, innovation_type, in_node=None, out_node=None):
        if in_node is None or out_node is None:
            return None

        if innovation_type == InnovationType.NODE:
            innovation = Innovation(self.next_innovation_num, innovation_type, in_node, out_node,
                                    node_id=self.next_node_id)
            self.next_node_id += 1
        elif innovation_type == InnovationType.CONNECTION:
            innovation = Innovation(self.next_innovation_num, innovation_type, in_node, out_node,
                                    connection_id=self.next_connection_id)
            self.next_connection_id += 1
        else:
            return None

        self.innovations.append(innovation)
        self.next_innovation_num += 1
        return innovation

    def get_innovation(self, innovation_type, in_node=None, out_node=None):
        innovation = self.exist_innovation(innovation_type, in_node, out_node)
        if innovation is None:
            innovation = self.create_innovation(innovation_type, in_node, out_node)
        return innovation
