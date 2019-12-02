from itertools import count
from random import random, choice

from NEAT.src.genes import NodeGene, ConnectionGene


class Genome:
    def __init__(self, genome_id):
        self.genome_id = genome_id
        self.connections = {}
        self.nodes = {}
        self.fitness = None

    def create_new(self, config):
        # add inputs
        for i in range(config.num_inputs):
            node_id = self.get_new_node_id(self.nodes)
            assert node_id not in self.nodes
            self.nodes[node_id] = NodeGene(node_id, 'INPUT')

        # add outputs
        for i in range(config.num_outputs):
            node_id = self.get_new_node_id(self.nodes)
            assert node_id not in self.nodes
            self.nodes[node_id] = NodeGene(node_id, 'OUTPUT')

        # add hidden nodes
        if config.num_hidden > 0:
            for i in range(config.num_hidden):
                node_id = self.get_new_node_id(self.nodes)
                assert node_id not in self.nodes
                self.nodes[node_id] = NodeGene(node_id, 'HIDDEN')

        # add connections
        if 'fs_neat' in config.initial_connection:
            # without hidden
            inputs = {node_id: node for node_id, node in self.nodes.items() if node.node_type == 'INPUT'}
            outputs = {node_id: node for node_id, node in self.nodes.items() if node.node_type == 'OUTPUT'}

            # randomly connect one input to all output nodes
            input_id = choice(list(inputs.keys()))
            for output_id in outputs.keys():
                connection = ConnectionGene((input_id, output_id))
                self.connections[connection.connection_id] = connection

        elif 'full' in config.initial_connection:
            inputs_ids = [i for i, node in self.nodes.items() if node.node_type == 'INPUT']
            outputs_ids = [i for i, node in self.nodes.items() if node.node_type == 'OUTPUT']
            hidden_ids = [i for i, node in self.nodes.items() if node.node_type == 'HIDDEN']
            connections = []
            if hidden_ids:
                for input_id in inputs_ids:
                    for hidden_id in hidden_ids:
                        connections.append((input_id, hidden_id))
                for hidden_id in hidden_ids:
                    for output_id in outputs_ids:
                        connections.append((hidden_id, output_id))
            if not hidden_ids:
                for input_id in inputs_ids:
                    for output_id in outputs_ids:
                        connections.append((input_id, output_id))

            for input_id, output_id in connections:
                connection = ConnectionGene((input_id, output_id))
                self.connections[connection.connection_id] = connection

        # maybe partial also later

    def crossover(self, genome1, genome2):
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))

        # choose more fit parent
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent2, parent1 = genome1, genome2

        # inherit nodes
        parent1_nodes = parent1.nodes
        parent2_nodes = parent2.nodes

        for node_id, node_gene1 in parent1_nodes.items():
            node_gene2 = parent2_nodes.get(node_id)
            assert node_id not in self.nodes
            if node_gene2 is None:
                # more gene in fittest parent - get it
                self.nodes[node_id] = node_gene1.copy()
            else:
                # combine from both
                self.nodes[node_id] = node_gene1.crossover(node_gene2)

        # inherit connections
        for connection_id, connection_gene1 in parent1.connections.items():
            connection_gene2 = parent2.connections.get(connection_id)
            if connection_gene2 is None:
                # excess or disjoint - copy connection from parent1
                self.connections[connection_id] = connection_gene1.copy()
            else:
                # combine from both
                self.connections[connection_id] = connection_gene1.crossover(connection_gene2)

    def mutate(self, config):
        if random() < config.node_add_prob:
            self.mutate_add_node()
        if random() < config.node_remove_prob:
            self.mutate_remove_node()
        if random() < config.conn_add_prob:
            self.mutate_add_connection()
        if random() < config.conn_remove_prob:
            self.mutate_remove_connection()

        # mutate nodes
        for node_gene in self.nodes.values():
            node_gene.mutate(config)

        # mutate connections
        for connection_gene in self.connections.values():
            connection_gene.mutate(config)

    def mutate_add_node(self):
        # random connection to divide
        connection_to_divide = choice(list(self.connections.values()))
        new_node_id = self.get_new_node_id(self.nodes)
        node_gene = NodeGene(new_node_id, 'HIDDEN')
        self.nodes[new_node_id] = node_gene

        # disable connection and create two new connections
        connection_to_divide.enabled = False
        i, o = connection_to_divide.connection_id

        conn1 = ConnectionGene((i, new_node_id))
        conn1.weight = 1.0
        self.connections[conn1.connection_id] = conn1

        conn2 = ConnectionGene((new_node_id, o))
        conn2.weight = connection_to_divide.weight
        self.connections[conn2.connection_id] = conn2

    def mutate_remove_node(self):
        available_nodes = [node_id for node_id, node in self.nodes.items() if node.node_type == 'HIDDEN']
        if not available_nodes:
            return -1

        delete_id = choice(available_nodes)

        connections_to_delete = set()
        for k, v in self.connections.items():
            if delete_id in v.connection_id:
                connections_to_delete.add(v.connection_id)

        for connection_id in connections_to_delete:
            del self.connections[connection_id]

        del self.nodes[delete_id]
        return delete_id

    def mutate_add_connection(self):
        possible_outputs = [node_id for node_id, node in self.nodes.items() if node.node_type != 'INPUT']
        out_node = choice(possible_outputs)

        possible_inputs = list(self.nodes.keys())
        in_node = choice(possible_inputs)

        # do not duplicate connections
        connection_id = (in_node, out_node)
        if connection_id in self.connections:
            return

        # do not connect outputs
        if self.nodes[in_node].node_type == 'OUTPUT' and self.nodes[out_node].node_type == 'OUTPUT':
            return

        # avoid creating cycles
        if self.creates_cycle(connection_id):
            return

        connection_gene = ConnectionGene(connection_id)
        self.connections[connection_id] = connection_gene

    def mutate_remove_connection(self):
        if self.connections:
            connection_id = choice(list(self.connections.keys()))
            del self.connections[connection_id]

    @staticmethod
    def get_new_node_id(node_dict):
        try:
            indexer = count(max(list(node_dict.keys())) + 1)
            new_id = next(indexer)
        except ValueError:
            new_id = 0
        assert new_id not in node_dict
        return new_id

    def creates_cycle(self, test_id):
        i, o = test_id
        if i == o:
            return True

        visited = {o}
        while True:
            num_added = 0
            for a, b in list(self.connections.keys()):
                if a in visited and b not in visited:
                    if b == i:
                        return True

                    visited.add(b)
                    num_added += 1

            if num_added == 0:
                return False

    def distance(self, other_genome, config):
        # count node genes distance
        # get node without input nodes
        my_nodes = {node_id: node for node_id, node in self.nodes.items() if node.node_type != 'INPUT'}
        other_nodes = {node_id: node for node_id, node in other_genome.nodes.items() if node.node_type != 'INPUT'}

        node_distance = 0.0
        if my_nodes or other_nodes:
            disjoint_nodes = 0
            for k2 in other_nodes.keys():
                if k2 not in my_nodes:
                    disjoint_nodes += 1

            for k1, node1 in my_nodes.items():
                node2 = other_nodes.get(k1)
                if node2 is None:
                    disjoint_nodes += 1
                else:
                    node_distance += node1.distance(node2, config)

            max_nodes_len = max(len(my_nodes), len(other_nodes))
            node_distance = (node_distance + (
                    disjoint_nodes * config.compatibility_disjoint_coefficient)) / max_nodes_len

        # count connection genes distance
        connection_distance = 0.0
        if self.connections or other_genome.connections:
            disjoint_connections = 0
            for k2 in other_genome.connections.keys():
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, conn1 in self.connections.items():
                conn2 = other_genome.connections.get(k1)
                if conn2 is None:
                    disjoint_connections += 1
                else:
                    connection_distance += conn1.distance(conn2, config)

            max_conn_len = max(len(self.connections), len(other_genome.connections))
            connection_distance = (connection_distance + (
                    config.compatibility_disjoint_coefficient * disjoint_connections)) / max_conn_len

        distance = node_distance + connection_distance
        return distance

    def size(self):
        number_of_enabled_conn = sum([1 for connection_gene in self.connections.values() if connection_gene.enabled])
        return len(self.nodes), number_of_enabled_conn

    def __str__(self):
        string = f'ID: {self.genome_id}\nFitness: {self.fitness}\nNodes:'
        for k, node_gene in self.nodes.items():
            string += f'\n\t{node_gene}'
        string += '\nConnections:'
        connections = list(self.connections.values())
        for conn in connections:
            string += f'\n\t{conn}'
        return string
