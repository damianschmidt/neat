import copy
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import graphviz


class Statistics:
    def __init__(self, task_name):
        self.task_name = task_name
        self.most_fit_genomes = []
        self.generation_statistics = []

    def draw_stats(self):
        generations = range(len(self.most_fit_genomes))
        best_fitnesses = [genome.fitness for genome in self.most_fit_genomes]
        average_fitnesses = np.array(self.get_fitness_stat(mean))
        stdev_fitnesses = np.array(self.get_fitness_stat(stdev))

        plt.plot(generations, average_fitnesses, 'b-', label='AVERAGE')
        plt.plot(generations, average_fitnesses - stdev_fitnesses, 'g-', label='-1 STDEV')
        plt.plot(generations, average_fitnesses + stdev_fitnesses, 'g-', label='+1 STDEV')
        plt.plot(generations, best_fitnesses, 'r-', label='BEST')

        plt.title('POPULATION STATISTICS')
        plt.ylabel('FITNESS')
        plt.xlabel('GENERATIONS')
        plt.grid()
        plt.legend()
        plt.savefig(f'{self.task_name}_population_stats.svg')
        plt.show()
        plt.close()

    def draw_species(self):
        species_size = self.get_species_sizes()
        generations = len(species_size)
        curves = np.array(species_size).T

        figure, ax = plt.subplots()
        ax.stackplot(range(generations), *curves)

        plt.title('SPECIES')
        plt.ylabel('SIZE OF SPECIES')
        plt.xlabel('GENERATIONS')
        plt.savefig(f'{self.task_name}_species.svg')
        plt.show()
        plt.close()

    def draw_genome(self, genome):
        node_settings = {
            'shape': 'circle',
            'font_size': '7',
            'height': '0.1',
            'width': '0.1'
        }

        dot = graphviz.Digraph(format='svg', node_attr=node_settings)

        inputs = set()
        input_nodes_ids = [node.node_id for node in genome.nodes.values() if node.node_type == 'INPUT']
        input_settings = {
            'style': 'filled',
            'fillcolor': 'red'
        }
        for node in input_nodes_ids:
            inputs.add(node)
            dot.node(name=str(node), _attributes=input_settings)

        outputs = set()
        output_nodes_ids = [node.node_id for node in genome.nodes.values() if node.node_type == 'OUTPUT']
        output_settings = {
            'style': 'filled',
            'fillcolor': 'blue'
        }
        for node in output_nodes_ids:
            outputs.add(node)
            dot.node(name=str(node), _attributes=output_settings)

        hidden = set()
        hidden_settings = {
            'style': 'filled',
            'fillcolor': 'green'
        }
        hidden_nodes_ids = [node.node_id for node in genome.nodes.values() if node.node_type == 'HIDDEN']
        for node in hidden_nodes_ids:
            hidden.add(node)
            dot.node(name=str(node), _attributes=hidden_settings)

        for connection in genome.connections.values():
            if connection.enabled:
                i, o = connection.connection_id
                connection_settings = {
                    'style': 'solid',
                    'color': 'black',
                    'penwidth': str(0.1 + abs(connection.weight / 5.0)),
                    'label': f'{connection.weight:.2f}'
                }
                dot.edge(str(i), str(o), _attributes=connection_settings)

        dot.render(f'{self.task_name}_genome', view=True)

        return dot

    def get_fitness_stat(self, function):
        stat = []
        for stats in self.generation_statistics:
            scores = []
            for species_stats in stats.values():
                scores.extend(species_stats.values())
            stat.append(function(scores))
        return stat

    def get_species_sizes(self):
        all_species = set()
        for gen_data in self.generation_statistics:
            all_species = all_species.union(gen_data.keys())

        max_species = max(all_species)
        species_counts = []
        for gen_data in self.generation_statistics:
            species = [len(gen_data.get(sid, [])) for sid in range(1, max_species + 1)]
            species_counts.append(species)

        return species_counts

    def post_evaluate(self, species, best_genome):
        self.most_fit_genomes.append(copy.deepcopy(best_genome))
        species_stats = {}
        for species_id, s in species.species.items():
            species_stats[species_id] = dict((k, v.fitness) for k, v in s.members.items())
        self.generation_statistics.append(species_stats)


