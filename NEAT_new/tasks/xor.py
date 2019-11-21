import os
import pickle

from NEAT_new.src.config import Config
from NEAT_new.src.genetic_algorithm import GeneticAlgorithm
from NEAT_new.src.network import Network

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def evaluate_genome(genomes):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        network = Network.create(genome)
        for xor_input, xor_output in zip(xor_inputs, xor_outputs):
            output = network.activate(xor_input)
            genome.fitness -= (output[0] - xor_output[0]) ** 2


def run():
    config = Config()
    population = GeneticAlgorithm(config)
    winner = population.run(evaluate_genome, 300)

    print(f'\nBEST GENOME:\n{winner}')

    dir_name = './results/'
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    with open('results/winner_xor.pkl', 'wb') as output:
        pickle.dump(winner, output, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    run()
