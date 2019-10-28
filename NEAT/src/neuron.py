import math


class Neuron:
    def __init__(self):
        self.output = 0.0
        self.inputs = []
        self.output_ids = []
        self.output_weights = []

    def add_output_connections(self, output_id, weight):
        self.output_ids.append(output_id)
        self.output_weights.append(weight)

    def add_input_connections(self):
        self.inputs.append(None)

    def calculate(self):
        total_sum = 0.0
        for i in self.inputs:
            total_sum += i

        self.output = self.sigmoid_activation_function(total_sum)
        return self.output

    def is_ready_to_calculate(self):
        is_ready = True
        for i in self.inputs:
            if not i:
                is_ready = False
                break
        return is_ready

    # Add input to the neuron in the first available slot
    def feed_input(self, new_input):
        found_slot = False
        for i in range(len(self.inputs)):
            if self.inputs[i] is None:
                self.inputs[i] = new_input
                found_slot = True
                break
        if not found_slot:
            raise RuntimeError(f'No input slot ready for input')

    def reset(self):
        self.inputs = [None for i in self.inputs]
        self.output = 0.0

    def sigmoid_activation_function(self, total_sum):
        result = 0.5
        if total_sum:
            # 4.9 from documentation
            result = 1.0 / (1.0 + math.exp(-4.9 * total_sum))
        return result
