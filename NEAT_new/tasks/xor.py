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

    def evaluate(self):
        pass


if __name__ == '__main__':
    task = XorTask()
