from NEAT.src.connection_gene import ConnectionGene
from NEAT.src.evaluator import Evaluator
from NEAT.src.genome import Genome
from NEAT.src.innovation_generator import InnovationGenerator
from NEAT.src.neural_network import NeuralNetwork
from NEAT.src.node_gene import NodeGene


class XOR(Evaluator):
    def __init__(self):
        self.input = [[0.0, 0.0, 1.0],
                      [1.0, 0.0, 1.0],
                      [0.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0]]
        self.correct_results = [0.0, 1.0, 1.0, 0.0]

        self.node_innovation = InnovationGenerator()
        self.conn_innovation = InnovationGenerator()

        self.genome = Genome()
        self.genome.add_node(NodeGene('SENSOR', self.node_innovation.get_innovation()))
        self.genome.add_node(NodeGene('SENSOR', self.node_innovation.get_innovation()))
        self.genome.add_node(NodeGene('SENSOR', self.node_innovation.get_innovation()))  # bias
        self.genome.add_node(NodeGene('OUTPUT', self.node_innovation.get_innovation()))

        self.genome.add_connection(ConnectionGene(0, 3, 1.0, True, self.conn_innovation.get_innovation()))
        self.genome.add_connection(ConnectionGene(1, 3, 1.0, True, self.conn_innovation.get_innovation()))
        self.genome.add_connection(ConnectionGene(2, 3, 1.0, True, self.conn_innovation.get_innovation()))
        super().__init__(self.genome, self.node_innovation, self.conn_innovation)

    def evaluate_genome(self, genome):
        total_distance = 0
        nn = NeuralNetwork(genome)

        for i in range(len(self.input)):
            inputs = [self.input[i][0], self.input[i][1], self.input[i][2]]
            outputs = nn.calculate(inputs)
            distance = abs(self.correct_results[i] - outputs[0])
            total_distance += distance
        return 4.0 - total_distance


if __name__ == '__main__':
    xor = XOR()
    for j in range(1000):
        xor.evaluate()
        print('Generation:', j)
        print('Highest fitness:', xor.highest_score)
        print('Amount of species:', len(xor.species))
        print('Connections in best performer:', len(xor.fittest_genome.dict_of_connections))
        print('Guesses:')

        net = NeuralNetwork(xor.fittest_genome)
        for k in range(len(xor.input)):
            net_inputs = [xor.input[k][0], xor.input[k][1], xor.input[k][2]]
            net_outputs = net.calculate(net_inputs)

            print(xor.input[k][0], xor.input[k][1], net_outputs[0])
        print()



