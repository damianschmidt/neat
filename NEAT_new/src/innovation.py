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
        pass

    def create_innovation(self, innovation_type, in_node=None, out_node=None):
        pass

    def get_innovation(self, innovation_type, in_node=None, out_node=None):
        pass
