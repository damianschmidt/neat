import numpy as np


class XorTask:
    @property
    def inputs_num(self):
        return np.shape(self.input_data)[1]

    @property
    def outputs_num(self):
        return np.shape(self.output_data)[1]

    def __init__(self):
        self.input_data = np.array([(0, 0), (0, 1), (1, 0), (1, 1)], dtype=float)
        self.output_data = np.array([(-1,), (1,), (1,), (-1,)], dtype=float)
        self.solved = False

    def evaluate(self, network):
        mse = 0.0
        for (input_datum, target) in zip(self.input_data, self.output_data):
            output = network.feed(input_datum)
            error = target - output
            error[abs(error) < 1e-100] = 0
            mse += (error ** 2).mean()

        rmse = np.sqrt(mse / len(self.input_data))
        fitness_score = 1 / (1 + rmse)
        solved = fitness_score > 0.99
        return [fitness_score, solved]


if __name__ == '__main__':
    task = XorTask()
