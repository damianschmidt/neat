from NEAT.src.neuron import Neuron


class NeuralNetwork:
    def __init__(self, genome):
        self.input = []
        self.output = []
        self.neurons = {}
        self.unprocessed = []

        # Add neurons initialization
        # Add nodes
        for node_id, node in genome.dict_of_nodes.items():
            neuron = Neuron()
            if node.node_type == 'SENSOR':
                neuron.add_input_connections()
                self.input.append(node_id)
            elif node.node_type == 'OUTPUT':
                self.output.append(node_id)
            self.neurons[node_id] = neuron

        # Add connections
        for connection_id, connection in genome.dict_of_connections.items():
            if not connection.expressed:
                continue
            in_node = self.neurons[connection.in_node]
            in_node.add_output_connections(connection.out_node, connection.weight)
            out_node = self.neurons[connection.out_node]
            out_node.add_input_connections()

    def calculate(self):
        pass
