from random import random


class ConnectionGene:
    def __init__(self, in_node, out_node, innovation_num, disabled=False, weight=None, recurrent=False):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = random() * 2 - 1 if weight is None else weight
        self.disabled = disabled
        self.recurrent = recurrent
        self.innovation_num = innovation_num
