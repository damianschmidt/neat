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

    def calculate(self, input_parameters):
        if len(input_parameters) != len(self.input):
            raise ValueError('Number of inputs must match number of input neurons in genome')

        self.reset_network()
        self.prepare_inputs(input_parameters)
        outputs = self.solve_network()
        return outputs

    def solve_network(self):
        loop = 0
        while len(self.unprocessed) > 0:
            loop += 1
            if loop > 1000:
                print('Can not solve the network')
                return None

            # iterate through the network and calculate neurons

        # copy output from output neurons
        outputs = []
        return outputs

    def prepare_inputs(self, input_parameters):
        for i in range(len(input_parameters)):
            input_neuron = self.neurons[self.input[i]]
            input_neuron.feed_input(input_parameters[i])
            input_neuron.calculate()  # ready for calculation cause inputs have only one input

            for j in range(len(input_neuron.output_ids)):
                receiver = self.neurons[input_neuron.output_ids[j]]
                receiver.feed_input(input_neuron.output * input_neuron.output_weights[j])

            self.unprocessed.remove(input_neuron)

    def reset_network(self):
        for neuron_id, neuron in self.neurons:
            neuron.reset()
        self.unprocessed = []
        self.unprocessed.extend(self.neurons.values())