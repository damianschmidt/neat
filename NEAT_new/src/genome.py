from random import randint, random

from NEAT_new.src.connection import ConnectionGene
from NEAT_new.src.innovation import InnovationType
from NEAT_new.src.node import NodeGene
from NEAT_new.src.node_type import NodeType


class Genome:
    def __init__(self, genome_id, innovation_set, nodes=None, connections=None, inputs_num=2, outputs_num=1):
        self.genome_id = genome_id
        self.innovation_set = innovation_set
        self.nodes = nodes
        self.connections = connections
        self.inputs_num = inputs_num
        self.outputs_num = outputs_num
        self.fitness = 0.0
        self.species_id = None

        # parameters
        self.tries_to_find_unconnected_nodes = 5

        self.weight_mutation_rate = 0.8
        self.reset_weight_rate = 0.1
        self.max_weight_perturbation = 0.5
        self.add_connection_rate = 0.05
        self.add_node_rate = 0.03
        self.activation_mutation_rate = 0.1
        self.max_activation_perturbation = 0.1

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

    def get_bias_input_output_nodes(self):
        return [x for x in self.nodes if x.type == NodeType.INPUT or x.type == NodeType.BIAS or x.type == NodeType.OUTPUT]

    def exist_node(self, node_id):
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def exist_connection(self, in_node, out_node):
        for con in self.connections:
            if con.in_node == in_node and con.out_node == out_node:
                return con
        return None

    def add_node(self):
        connection = None
        while connection is None:
            temp_connection = self.connections[randint(0, len(self.connections) - 1)]
            if not temp_connection.disabled and not temp_connection.recurrent and self.exist_node(
                    temp_connection.in_node).type != NodeType.BIAS:
                connection = temp_connection

        in_node = self.exist_node(connection.in_node)
        out_node = self.exist_node(connection.out_node)
        innovation = self.innovation_set.get_innovation(InnovationType.NODE, in_node, out_node)

        node = self.exist_node(innovation.node_id)
        if node is None:
            node = NodeGene(innovation.node_id, NodeType.HIDDEN)
            self.nodes.append(node)

            innovation1 = self.innovation_set.get_innovation(InnovationType.CONNECTION, in_node, node.node_id)
            con1 = ConnectionGene(in_node, node.node_id, innovation1.innovation_num, weight=1.0)
            self.connections.append(con1)

            innovation2 = self.innovation_set.get_innovation(InnovationType.CONNECTION, node.node_id, out_node)
            con2 = ConnectionGene(node.node_id, out_node, innovation2.innovation_num, weight=connection.weight)
            self.connections.append(con2)

            connection.disabled = True
            return (node, con1, con2)
        else:
            return None

    def add_connection(self):
        node1, node2 = None, None

        if node1 is None:
            for _ in range(self.tries_to_find_unconnected_nodes):
                temp_node1 = self.nodes[randint(0, len(self.nodes) - 1)]
                temp_node2 = self.nodes[randint(1 + self.inputs_num, len(self.nodes) - 1)]

                if not self.exist_connection(temp_node1.node_id, temp_node2.node_id):
                    node1 = temp_node1
                    node2 = temp_node2

        if node1 is None or node2 is None:
            return None

        innovation = self.innovation_set.get_innovation(InnovationType.CONNECTION, node1.node_id, node2.node_id)
        weight = random() * 2 - 1
        connection = ConnectionGene(node1.node_id, node2.node_id, innovation.innovation_num, weight=weight)
        return connection

    def mutation(self):
        # add node
        if random() < self.add_node_rate:
            self.add_node()

        # add connection
        if random() < self.add_connection_rate:
            self.add_connection()

        # mutate weights
        for con in self.connections:
            if random() < self.weight_mutation_rate:
                if random() < self.reset_weight_rate:
                    con.weight = random() * 2 - 1
                else:
                    con.weight += (random() * 2 - 1) * self.max_weight_perturbation

        # mutate activation response
        for node in self.nodes:
            if random() > self.activation_mutation_rate:
                node.activation_response += (random() * 2 - 1) * self.max_activation_perturbation

    def __str__(self):
        string = f'Genome {self.genome_id} {self.fitness} \n' \
                 f'Inputs {self.inputs_num} \n' \
                 f'Outputs {self.outputs_num} \n' \
                 f'Bias nodes {len(self.get_bias_nodes())} \n' \
                 f'Input nodes {len(self.get_input_nodes())} \n' \
                 f'Hidden nodes {len(self.get_hidden_nodes())} \n' \
                 f'Output nodes {len(self.get_output_nodes())} \n'
        return string
