from NEAT.src.connection_gene import ConnectionGene
from NEAT.src.evaluator import Evaluator
from NEAT.src.genome import Genome
from NEAT.src.innovation_generator import InnovationGenerator
from NEAT.src.neural_network import NeuralNetwork
from NEAT.src.node_gene import NodeGene


class XOR(Evaluator):
    def __init__(self):
        self.input = [[0.0, 0.0, 0.5],
                      [1.0, 0.0, 0.5],
                      [0.0, 1.0, 0.5],
                      [1.0, 1.0, 0.5]]
        self.correct_results = [0.0, 0.0, 1.0, 1.0]

        self.node_innovation = InnovationGenerator()
        self.conn_innovation = InnovationGenerator()

        self.genome = Genome()
        self.genome.add_node(NodeGene('SENSOR', self.node_innovation.get_innovation()))
        self.genome.add_node(NodeGene('SENSOR', self.node_innovation.get_innovation()))
        self.genome.add_node(NodeGene('SENSOR', self.node_innovation.get_innovation()))  # bias
        self.genome.add_node(NodeGene('HIDDEN', self.node_innovation.get_innovation()))
        self.genome.add_node(NodeGene('OUTPUT', self.node_innovation.get_innovation()))

        self.genome.add_connection(ConnectionGene(0, 3, 1.0, True, self.conn_innovation.get_innovation()))
        self.genome.add_connection(ConnectionGene(1, 3, 1.0, True, self.conn_innovation.get_innovation()))
        self.genome.add_connection(ConnectionGene(2, 3, 1.0, True, self.conn_innovation.get_innovation()))
        self.genome.add_connection(ConnectionGene(0, 4, 1.0, True, self.conn_innovation.get_innovation()))
        self.genome.add_connection(ConnectionGene(1, 4, 1.0, True, self.conn_innovation.get_innovation()))
        self.genome.add_connection(ConnectionGene(2, 4, 1.0, True, self.conn_innovation.get_innovation()))
        self.genome.add_connection(ConnectionGene(3, 4, 1.0, True, self.conn_innovation.get_innovation()))
        super().__init__(self.genome, self.node_innovation, self.conn_innovation)

    def evaluate_genome(self, genome):
        total_distance = 0
        nn = NeuralNetwork(genome)

        for i in range(len(self.input)):
            inputs = [self.input[i][0], self.input[i][1], self.input[i][2]]
            output = None
            try:
                output = nn.calculate(inputs)
            except Exception as e:
                print(e)

            if output is None:
                print('Network failed!')

            guess = output
            distance = abs(self.correct_results[i] - guess)
            total_distance += distance
        return 4.0 - total_distance


if __name__ == '__main__':
    xor = XOR()
    # xor.evaluate()
