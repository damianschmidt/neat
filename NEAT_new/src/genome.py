from NEAT_new.src.node_type import NodeType


class Genome:
    def __init__(self, genome_id, innovation_db, nodes=None, connections=None, inputs_num=2, outputs_num=1):
        self.genome_id = genome_id
        self.innovation_db = innovation_db
        self.nodes = nodes
        self.connections = connections
        self.inputs_num = inputs_num
        self.outputs_num = outputs_num
        self.fitness = 0.0
        self.species_id = None

        # parameters
        self.mutation_rate = 0.8
        self.add_connection_rate = 0.05
        self.add_node_rate = 0.03

        if nodes is not None:
            self.nodes.sort(key=lambda x: x.node_id)

        # create genome from phenotype
        # crate genome based on number of inputs and outputs

    def get_input_nodes(self):
        return [x for x in self.nodes if x.node_type == NodeType.INPUT]

    def get_output_nodes(self):
        return [x for x in self.nodes if x.node_type == NodeType.OUTPUT]

    def get_bias_nodes(self):
        return [x for x in self.nodes if x.node_type == NodeType.BIAS]

    def get_hidden_nodes(self):
        return [x for x in self.nodes if x.node_type == NodeType.HIDDEN]

    def exist_connection(self, in_node, out_node):
        for con in self.connections:
            if con.in_node == in_node and con.out_node == out_node:
                return con
        return None

    def add_connection(self):
        pass

    def add_node(self):
        pass

    def mutation(self):
        pass
