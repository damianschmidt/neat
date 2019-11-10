import time

import numpy as np

from NEAT_new.src.genetic_algorithm import GeneticAlgorithm


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
        return fitness_score, solved


if __name__ == '__main__':
    task = XorTask()

    # only benchmarks
    generations = np.array([])
    start_time = time.time()
    for i in range(1):  # 50 tries of solving problem
        algorithm = GeneticAlgorithm(task)
        print('Try:', i)
        for j in range(500):  # 500 epochs
            if algorithm.evaluator():  # if solved
                generations = np.append(generations, algorithm.generation)
                duration = time.time() - start_time
                print(f'Average generations: {generations.mean()}\n'
                      f'Std: {generations.std()}\n'
                      f'Size: {generations.size()}\n'
                      f'Time: {duration}')
                break
