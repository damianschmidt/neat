from NEAT_new.src.config import Config
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


def run(configuration):
    pass


if __name__ == '__main__':
    config = Config()
    run(config)
