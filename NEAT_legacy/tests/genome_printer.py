from graphics import *
from random import randint

from NEAT_legacy.src.genome import Genome
from NEAT_legacy.src.node_gene import NodeGene
from NEAT_legacy.src.connection_gene import ConnectionGene


class GenomePrinter:
    def __init__(self):
        self.window_height = 700
        self.window_width = 700
        self.node_size = 20

    def crossover_printer(self, genome):
        win = GraphWin('Test Crossover', self.window_width, self.window_height)
        win.setBackground('white')
        node_positions = {}

        self.draw_nodes(genome, node_positions, win)
        self.draw_nodes_id(node_positions, win)
        self.draw_connections(genome, node_positions, win)

        win.getMouse()

    def draw_connections(self, genome, node_positions, win):
        for connection in genome.dict_of_connections.values():
            if not connection.expressed:
                continue

            in_node = node_positions[connection.in_node]
            out_node = node_positions[connection.out_node]

            line_vector = Point(int((out_node.x - in_node.x) * 1.01), int((out_node.y - in_node.y) * 1.01))
            connection_line = Line(Point(in_node.x, in_node.y),
                                   Point(in_node.x + line_vector.x, in_node.y + line_vector.y))
            connection_line.draw(win)

            vector_direction = Circle(Point(out_node.x * 1.02, out_node.y * 1.02), 5)
            vector_direction.setFill('red')
            vector_direction.draw(win)

            self.draw_connection_weights(connection, in_node, out_node, win)

    def draw_connection_weights(self, connection, in_node, out_node, win):
        msg_x = (out_node.x + in_node.x) / 2
        msg_y = (out_node.y + in_node.y) / 2
        msg_connection_weight = Text(Point(msg_x, msg_y), connection.weight)
        msg_connection_weight.setTextColor('black')
        msg_connection_weight.draw(win)

    def draw_nodes_id(self, node_positions, win):
        for key, node in node_positions.items():
            msg_node_id = Text(node, key)
            msg_node_id.setTextColor('white')
            msg_node_id.draw(win)

    def draw_nodes(self, genome, node_positions, win):
        input_counter = 0
        output_counter = 0
        for node in genome.dict_of_nodes.values():
            if node.node_type == 'SENSOR':
                x = (self.window_width / (self.count_nodes_by_types(genome, 'SENSOR') + 1)) * (input_counter + 1)
                y = self.window_height - 2 * self.node_size
                input_counter += 1
                node_positions[node.node_id] = Point(x, y)

                node_circle = Circle(Point(x, y), self.node_size)
                node_circle.setFill('blue')
                node_circle.draw(win)
            elif node.node_type == 'HIDDEN':
                x = randint(0, self.window_width - 2 * self.node_size) + self.node_size
                y = randint(3 * self.node_size, self.window_height - 4 * self.node_size)
                node_positions[node.node_id] = Point(x, y)

                node_circle = Circle(Point(x, y), self.node_size)
                node_circle.setFill('blue')
                node_circle.draw(win)
            else:
                x = (self.window_width / (self.count_nodes_by_types(genome, 'OUTPUT') + 1)) * (output_counter + 1)
                y = self.node_size
                output_counter += 1
                node_positions[node.node_id] = Point(x, y)

                node_circle = Circle(Point(x, y), self.node_size)
                node_circle.setFill('blue')
                node_circle.draw(win)

    def count_nodes_by_types(self, genome, node_type):
        counter = 0
        for node in genome.dict_of_nodes.values():
            if node.node_type == node_type:
                counter += 1

        return counter


if __name__ == '__main__':
    genome_printer = GenomePrinter()

    parent_genome1 = Genome()
    parent_genome1.dict_of_nodes[1] = NodeGene('SENSOR', 1)
    parent_genome1.dict_of_nodes[2] = NodeGene('SENSOR', 2)
    parent_genome1.dict_of_nodes[3] = NodeGene('SENSOR', 3)
    parent_genome1.dict_of_nodes[4] = NodeGene('OUTPUT', 4)
    parent_genome1.dict_of_nodes[5] = NodeGene('HIDDEN', 5)
    parent_genome1.dict_of_connections[1] = ConnectionGene(1, 4, 0.5, True, 1)
    parent_genome1.dict_of_connections[2] = ConnectionGene(2, 4, 0.5, False, 2)
    parent_genome1.dict_of_connections[3] = ConnectionGene(3, 4, 0.5, True, 3)
    parent_genome1.dict_of_connections[4] = ConnectionGene(2, 5, 0.5, True, 4)
    parent_genome1.dict_of_connections[5] = ConnectionGene(5, 4, 0.5, True, 5)
    parent_genome1.dict_of_connections[8] = ConnectionGene(1, 5, 0.5, True, 8)
    genome_printer.crossover_printer(parent_genome1)

    parent_genome2 = Genome()
    parent_genome2.dict_of_nodes[1] = NodeGene('SENSOR', 1)
    parent_genome2.dict_of_nodes[2] = NodeGene('SENSOR', 2)
    parent_genome2.dict_of_nodes[3] = NodeGene('SENSOR', 3)
    parent_genome2.dict_of_nodes[4] = NodeGene('OUTPUT', 4)
    parent_genome2.dict_of_nodes[5] = NodeGene('HIDDEN', 5)
    parent_genome2.dict_of_nodes[6] = NodeGene('HIDDEN', 6)
    parent_genome2.dict_of_connections[1] = ConnectionGene(1, 4, 0.5, True, 1)
    parent_genome2.dict_of_connections[2] = ConnectionGene(2, 4, 0.5, False, 2)
    parent_genome2.dict_of_connections[3] = ConnectionGene(3, 4, 0.5, True, 3)
    parent_genome2.dict_of_connections[4] = ConnectionGene(2, 5, 0.5, True, 4)
    parent_genome2.dict_of_connections[5] = ConnectionGene(5, 4, 0.5, False, 5)
    parent_genome2.dict_of_connections[6] = ConnectionGene(5, 6, 0.5, True, 6)
    parent_genome2.dict_of_connections[7] = ConnectionGene(6, 4, 0.5, True, 7)
    parent_genome2.dict_of_connections[9] = ConnectionGene(3, 5, 0.5, True, 9)
    parent_genome2.dict_of_connections[10] = ConnectionGene(1, 6, 0.5, True, 10)
    genome_printer.crossover_printer(parent_genome2)

    genome = Genome()
    child_genome = genome.crossover(parent_genome2, parent_genome1)
    genome_printer.crossover_printer(child_genome)
