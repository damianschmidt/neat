import os
import pickle

from NEAT.src.config import Config
from NEAT.src.genetic_algorithm import GeneticAlgorithm
from NEAT.src.network import Network
from NEAT.src.statistics import Statistics

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
    try:
        with open('results/genomes/xor/winner_xor_fs_neat.pkl', 'rb') as input_file:
            default_genome = pickle.load(input_file)
    except FileNotFoundError:
        print('No previous winner data! Create new genome set')
        default_genome = None

    config = Config()
    stats = Statistics(task_name='xor_fs_neat')
    population = GeneticAlgorithm(config, default_genome, stats)

    winner = population.run(evaluate_genome, 300)

    print(f'\nBEST GENOME:\n{winner}')

    dir_name = './results/'
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    with open('results/genomes/xor/winner_xor_fs_neat.pkl', 'wb') as output:
        pickle.dump(winner, output, protocol=pickle.HIGHEST_PROTOCOL)

    stats.draw_genome(winner)
    stats.draw_stats()
    stats.draw_species()


if __name__ == '__main__':
    run()
