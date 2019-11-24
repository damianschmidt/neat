import math
from statistics import median, mean


class Network:
    def __init__(self, inputs, outputs, node_evals):
        self.inputs = inputs
        self.outputs = outputs
        self.node_evals = node_evals
        self.values = {key: 0.0 for key in inputs + outputs}

    def activate(self, inputs):
        if len(self.inputs) != len(inputs):
            raise RuntimeError('Wrong number of inputs!')

        for i_id, i_val in zip(self.inputs, inputs):
            self.values[i_id] = i_val

        for node, activation_function, aggregation_function, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = aggregation_function(node_inputs)
            self.values[node] = activation_function(bias + response * s)

        return [self.values[i] for i in self.outputs]

    @staticmethod
    def create(genome):
        connections = [connection_gene.connection_id for connection_gene in genome.connections.values() if
                       connection_gene.enabled]

        input_ids = [node_id for node_id, node in genome.nodes.items() if node.node_type == 'INPUT']
        output_ids = [node_id for node_id, node in genome.nodes.items() if node.node_type == 'OUTPUT']
        layers = Network.layers(input_ids, output_ids, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                for conn_id in connections:
                    in_node, out_node = conn_id
                    if out_node == node:
                        connection_gene = genome.connections[conn_id]
                        inputs.append((in_node, connection_gene.weight))

                node_gene = genome.nodes[node]
                if node_gene.aggregation == 'sum':
                    aggregation_function = sum
                elif node_gene.aggregation == 'max':
                    aggregation_function = max
                elif node_gene.aggregation == 'median':
                    aggregation_function = median
                elif node_gene.aggregation == 'mean':
                    aggregation_function = mean
                else:
                    aggregation_function = None

                if node_gene.activation == 'sigmoid':
                    activation_function = Network.sigmoid
                else:
                    activation_function = None

                node_evals.append(
                    (node, activation_function, aggregation_function, node_gene.bias, node_gene.response, inputs))

        return Network(input_ids, output_ids, node_evals)

    @staticmethod
    def sigmoid(x):
        x = max(-60.0, min(60.0, 5.0 * x))
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def sigmoid_paper(x):
        # 4.9 from documentation
        return 1.0 / (1.0 + math.exp(-4.9 * x))

    @staticmethod
    def tanh(x):
        x = max(-60.0, min(60.0, 2.5 * x))
        return math.tanh(x)

    @staticmethod
    def relu(z):
        return z if z > 0.0 else 0.0

    @staticmethod
    def layers(inputs, outputs, connections):
        required = set(outputs)
        s = set(outputs)
        while True:
            # find nodes not in S whose output is consumed by a node in s
            t = set(a for (a, b) in connections if b in s and a not in s)

            if not t:
                break

            layer_nodes = set(x for x in t if x not in inputs)
            if not layer_nodes:
                break

            required = required.union(layer_nodes)
            s = s.union(t)

        layers = []
        s = set(inputs)
        while True:
            # find candidate nodes c for the next layer.  These nodes should connect
            # a node in s to a node not in s
            c = set(b for (a, b) in connections if a in s and b not in s)
            t = set()
            for n in c:
                if n in required and all(a in s for (a, b) in connections if b == n):
                    t.add(n)

            if not t:
                break

            layers.append(t)
            s = s.union(t)

        return layers
