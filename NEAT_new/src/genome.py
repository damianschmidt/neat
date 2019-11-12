import time
from math import sqrt
from random import randint, random, choice

import numpy as np

from NEAT_new.src.connection import ConnectionGene
from NEAT_new.src.innovation import InnovationType
from NEAT_new.src.node import NodeGene
from NEAT_new.src.node import NodeType
from NEAT_new.src.network import Network


class Genome:
    def __init__(self, genome_id, innovation_set, nodes=None, connections=None, inputs_num=2, outputs_num=1,
                 network=None):
        self.genome_id = genome_id
        self.innovation_set = innovation_set
        self.nodes = nodes
        self.connections = connections
        self.inputs_num = inputs_num
        self.outputs_num = outputs_num
        self.fitness = 0.0
        self.species_id = None
        self.adjusted_fitness = 0.0
        self.network = None
        self.solved = False

        # parameters
        self.tries_to_find_unconnected_nodes = 5
        self.tries_to_find_old_connection = 5
        self.weight_mutation_rate = 0.8
        self.reset_weight_rate = 0.1
        self.max_weight_perturbation = 0.5
        self.add_connection_rate = 0.07
        self.add_node_rate = 0.03
        self.activation_mutation_rate = 0.1
        self.max_activation_perturbation = 0.1
        self.feature_selection_neat = True
        self.chance_to_add_recurrent_connection = 0.05

        # weights
        self.stdev_weight = 2.0
        self.stdev_mutate_weight = 1.5

        if nodes is not None:
            self.nodes.sort(key=lambda x: x.node_id)

        # create genome from network
        if network is not None:
            inputs_num = 0
            outputs_num = 0
            next_node_id = 0

            self.nodes = []
            for node in network.nodes:
                new_node = NodeGene(node.node_id, node.node_type, node.pos_x, node.pos_y, node.activation_response)
                if new_node.node_type == NodeType.INPUT:
                    inputs_num += 1
                elif new_node.node_type == NodeType.OUTPUT:
                    outputs_num += 1
                self.nodes.append(new_node)
                next_node_id = max(next_node_id, node.node_id)
            next_node_id += 1
            innovation_set.next_node_id = max(innovation_set.next_node_id, next_node_id)

            self.connections = []
            for connection in network.connections:
                in_node = connection.in_node.node_id
                out_node = connection.out_node.node_id
                innovation = innovation_set.get_innovation(InnovationType.CONNECTION, in_node=in_node,
                                                           out_node=out_node)
                connection = ConnectionGene(in_node, out_node, innovation.innovation_num, weight=connection.weight)
                self.connections.append(connection)
            return

        # crate genome based on number of inputs and outputs
        if self.nodes is None:
            in_pos_x = 1. / (inputs_num + 1)
            out_pos_x = 1. / outputs_num
            next_node_id = 0
            self.nodes = []
            # add bias gene
            self.nodes.append(NodeGene(next_node_id, NodeType.BIAS, 0.5 * in_pos_x, 0.0))
            next_node_id += 1
            # add input nodes
            for i in range(inputs_num):
                self.nodes.append(NodeGene(next_node_id, NodeType.INPUT, (i + 1 + 0.5) * in_pos_x, 0.0))
                next_node_id += 1
            # add output nodes
            for i in range(outputs_num):
                self.nodes.append(NodeGene(next_node_id, NodeType.OUTPUT, (i + 0.5) * out_pos_x, 1.0))
                next_node_id += 1
            innovation_set.next_node_id = max(innovation_set.next_node_id, next_node_id)

        if self.connections is None:
            # add connections
            self.connections = []
            if self.feature_selection_neat:
                # connect random one input to one output
                random_input = choice(self.get_input_nodes())
                random_output = choice(self.get_output_nodes())
                innovation = innovation_set.get_innovation(InnovationType.CONNECTION, in_node=random_input.node_id,
                                                           out_node=random_output.node_id)
                weight = np.random.normal(0, self.stdev_weight)  # random() * 2 - 1
                self.connections.append(
                    ConnectionGene(random_input.node_id, random_output.node_id, innovation.innovation_num,
                                   weight=weight))
            else:
                # fully connected genome
                for i in self.get_bias_input_nodes():
                    for o in self.get_output_nodes():
                        innovation = innovation_set.get_innovation(InnovationType.CONNECTION, in_node=i.node_id,
                                                                   out_node=o.node_id)
                        weight = np.random.normal(0, self.stdev_weight)  # random() * 2 - 1
                        self.connections.append(
                            ConnectionGene(i.node_id, o.node_id, innovation.innovation_num, weight=weight))

    def get_input_nodes(self):
        return [x for x in self.nodes if x.node_type == NodeType.INPUT]

    def get_output_nodes(self):
        return [x for x in self.nodes if x.node_type == NodeType.OUTPUT]

    def get_bias_nodes(self):
        return [x for x in self.nodes if x.node_type == NodeType.BIAS]

    def get_hidden_nodes(self):
        return [x for x in self.nodes if x.node_type == NodeType.HIDDEN]

    def get_bias_input_nodes(self):
        return [x for x in self.nodes if x.node_type == NodeType.INPUT or x.node_type == NodeType.BIAS]

    def get_bias_input_output_nodes(self):
        return [x for x in self.nodes if
                x.node_type == NodeType.INPUT or x.node_type == NodeType.BIAS or x.node_type == NodeType.OUTPUT]

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
        # if genome is less then 5 hidden neurons, it is considered to be to small to select a link at random
        size_threshold = self.inputs_num + self.outputs_num + 5
        if len(self.connections) < size_threshold:
            for _ in range(self.tries_to_find_old_connection):
                temp_connection = self.connections[
                    randint(0, len(self.connections) - 1 - int(sqrt(len(self.connections) - 1)))]
                if not temp_connection.disabled and not temp_connection.recurrent and self.exist_node(
                        temp_connection.in_node).node_type != NodeType.BIAS:
                    connection = temp_connection
                    break
                if connection is None:
                    return
        else:
            while connection is None:
                temp_connection = self.connections[randint(0, len(self.connections) - 1)]
                if not temp_connection.disabled and not temp_connection.recurrent and self.exist_node(
                        temp_connection.in_node).node_type != NodeType.BIAS:
                    connection = temp_connection

        in_node = self.exist_node(connection.in_node)
        out_node = self.exist_node(connection.out_node)
        split_x = (in_node.pos_x + out_node.pos_x) / 2
        split_y = (in_node.pos_y + out_node.pos_y) / 2
        recurrent = in_node.pos_y > out_node.pos_y
        innovation = self.innovation_set.get_innovation(InnovationType.NODE, in_node, out_node)

        node = self.exist_node(innovation.node_id)
        if node is None:
            node = NodeGene(innovation.node_id, NodeType.HIDDEN, split_x, split_y)
            self.nodes.append(node)

            innovation1 = self.innovation_set.get_innovation(InnovationType.CONNECTION, in_node, node.node_id)
            con1 = ConnectionGene(in_node.node_id, node.node_id, innovation1.innovation_num, weight=1.0,
                                  recurrent=recurrent)
            self.connections.append(con1)

            innovation2 = self.innovation_set.get_innovation(InnovationType.CONNECTION, node.node_id, out_node)
            con2 = ConnectionGene(node.node_id, out_node.node_id, innovation2.innovation_num, weight=connection.weight,
                                  recurrent=recurrent)
            self.connections.append(con2)
            connection.disabled = True

            # why I return sth?
            return node, con1, con2
        else:
            return None

    def add_connection(self):
        node1, node2 = None, None
        recurrent = False

        if node1 is None:
            for _ in range(self.tries_to_find_unconnected_nodes):
                temp_node1 = self.nodes[randint(0, len(self.nodes) - 1)]
                temp_node2 = self.nodes[randint(1 + self.inputs_num, len(self.nodes) - 1)]

                if not self.exist_connection(temp_node1.node_id, temp_node2.node_id):
                    if temp_node1.pos_y >= temp_node2.pos_y:
                        if random() < self.chance_to_add_recurrent_connection:
                            node1 = temp_node1
                            node2 = temp_node2
                            recurrent = True
                            break
                    else:
                        node1 = temp_node1
                        node2 = temp_node2
                        break

        if node1 is None or node2 is None:
            # print('test')
            return None
        innovation = self.innovation_set.get_innovation(InnovationType.CONNECTION, node1.node_id, node2.node_id)
        weight = np.random.normal(0, self.stdev_weight)
        connection = ConnectionGene(node1.node_id, node2.node_id, innovation.innovation_num, weight=weight,
                                    recurrent=recurrent)
        self.connections.append(connection)
        return connection  # why I return sth?

    def mutation(self):
        # add node
        if random() < self.add_node_rate:
            self.add_node()

        # add connection
        if random() < self.add_connection_rate:
            # print('pre', self.connections)
            self.add_connection()
            # print('after', self.connections)
            # time.sleep(5)

        # mutate weights
        for con in self.connections:
            if random() < self.weight_mutation_rate:
                if random() < self.reset_weight_rate:
                    con.weight = np.random.normal(0, self.stdev_weight)
                else:
                    # con.weight += (random() * 2 - 1) * self.max_weight_perturbation
                    con.weight += np.random.normal(0, self.stdev_mutate_weight)
                    con.weight = np.clip(con.weight, -10., 10.)

        for node in self.nodes:
            if random() > self.activation_mutation_rate:
                node.activation_response += (random() * 2 - 1) * self.max_activation_perturbation

    def create_network(self):
        self.network = Network(self)
        return self.network

    def __str__(self):
        string = f'Genome       {self.genome_id} {self.fitness} \n' \
                 f'Inputs       {self.inputs_num} \n' \
                 f'Outputs      {self.outputs_num} \n' \
                 f'Bias nodes   {len(self.get_bias_nodes())} \n' \
                 f'Input nodes  {len(self.get_input_nodes())} \n' \
                 f'Hidden nodes {len(self.get_hidden_nodes())} \n' \
                 f'Output nodes {len(self.get_output_nodes())} \n' \
                 f'Connections  {self.connections} \n' \
                 f'Nodes:       {self.nodes} \n'
        return string
