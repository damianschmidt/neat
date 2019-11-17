import time

from NEAT_new.src.config import Config
from NEAT_new.src.genetic_algorithm import GeneticAlgorithm
from NEAT_new.src.network import Network
from NEAT_new.src.report import Reporter

xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
xor_outputs = [(0.0,), (1.0,), (1.0,), (0.0,)]


def evaluate_genome(genomes):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        network = Network.create(genome)
        for xor_input, xor_output in zip(xor_inputs, xor_outputs):
            output = network.activate(xor_input)
            # x = [x for x in genome.nodes.values() if x.node_type == 'HIDDEN']
            # if len(x) == 0:
            #     print(output[0], xor_output)
            genome.fitness -= (output[0] - xor_output[0]) ** 2

        # x = [x for x in genome.nodes.values() if x.node_type == 'HIDDEN']
        # if len(x) == 0:
        #     print(genome.fitness)
        #     print(genome.nodes)
        #     print(genome.connections)
        #     time.sleep(10)


def run():
    config = Config()
    population = GeneticAlgorithm(config)
    population.reporters.add(Reporter(True))
    winner = population.run(evaluate_genome, 300)

    print(f'\nBEST GENOME:\n{winner}')


if __name__ == '__main__':
    run()
