class Network:
    def __init__(self, inputs, outputs, node_evals):
        self.inputs = inputs
        self.outputs = outputs
        self.node_evals = node_evals
        self.values = {key: 0.0 for key in inputs + outputs}

    def activate(self, inputs):
        if len(self.inputs) != len(inputs):
            raise RuntimeError('Wrong number of inputs!')

        for k, v in zip (self.inputs, inputs):
            self.values[v] = k

        for node, activation_function, aggregation_function, bias, response, links in self.node_evals:
            inputs = []
            for i, w in links:
                inputs.append(self.values[i] * w)
            s = aggregation_function(inputs)
            self.values[node] = activation_function(bias + response * s)

        return [self.values[i] for i in self.outputs]
