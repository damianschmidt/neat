import os
import shutil
from copy import deepcopy
from random import randint, choice

from NEAT_new.src.genome import Genome
from NEAT_new.src.innovation import InnovationSet


class GeneticAlgorithm:
    def __init__(self, task):
        self.genomes = []
        self.species = []
        self.bests = []
        self.best_ever = None
        self.innovation_set = InnovationSet()
        self.task = task
        self.generation = 0
        self.pool = None

        self.next_genome_id = 0
        self.next_species_id = 0

        # parameters
        self.population_size = 100
        self.number_generation_allowed_to_not_improve = 20
        self.crossover_rate = 0.7
        self.survival_rate = 0.3

    def initialize(self):
        if not hasattr(self.task, 'name'):
            self.task.name = type(self.task).__name__

        self.results_path = f'./results/{self.task.name}'
        if os.path.exists(self.results_path):
            shutil.rmtree(self.results_path)
        os.mkdir(self.results_path)

        self.statistics = {
            'species': [],
            'generation': []
        }

    def evaluator(self):
        if not hasattr(self, 'statistics'):
            self.initialize()

        total_average = sum(s.average_fitness for s in self.species)
        for s in self.species:
            s.spawns_required = int(round(self.population_size * s.average_fitness / total_average))

        # remove stagnated species and species with no offsprings
        species = []
        for s in self.species:
            if s.generations_not_improved < self.number_generation_allowed_to_not_improve and s.spawns_required > 0:
                species.append(s)
        self.species[:] = species

        # reproduction
        for s in self.species:
            k = max(1, int(round(len(s.members) * self.survival_rate)))
            pool = s.members[:k]
            s.members[:] = []

            while len(s.members) < s.spawns_required:
                n = self.population_size / 5
                g1 = self.tournament_selection(pool, n)
                g2 = self.tournament_selection(pool, n)
                child = self.crossover(g1, g2, self.next_genome_id)
                child.mutation()
                s.add_member(child)

        self.genomes[:] = []
        for s in self.species:
            self.genomes.extend(s.members)
            s.members[:] = []
            s.age += 1

        # create basic population / initial birth
        # create new phenotypes and evaluate network
        # sort genomes by fitness
        # update best
        # assign genomes to species
        # remove empty species
        # adjust compatibility_threshold
        # sort members by fitness, adjust fitness, average_fitness and max_fitness
        # visualize

    @staticmethod
    def tournament_selection(genomes, number_to_compare):
        champion = None
        for _ in range(number_to_compare):
            g = genomes[randint(0, len(genomes) - 1)]
            if champion is None or g.fitness > champion.fitness:
                champion = g
        return champion

    @staticmethod
    def crossover(genome1, genome2, child_id):
        n_genome1 = len(genome1.connections)
        n_genome2 = len(genome2.connections)

        if genome1.fitness == genome2.fitness:
            if n_genome1 == n_genome2:
                better_genome = (genome1, genome2)[randint(0, 1)]
            elif n_genome1 < n_genome2:
                better_genome = genome1
            else:
                better_genome = genome2
        elif genome1.fitness > genome2.fitness:
            better_genome = genome1
        else:
            better_genome = genome2

        child_nodes = []
        child_connections = []

        # iterate through parents genes
        it_genome1, it_genome2 = 0, 0
        node_ids = set()
        while it_genome1 < n_genome1 or it_genome2 < n_genome2:
            genome1_gene = genome1.connections[it_genome1] if it_genome1 < n_genome1 else None
            genome2_gene = genome2.connections[it_genome2] if it_genome2 < n_genome2 else None
            selected_gene = None
            if genome1_gene and genome2_gene:
                if genome1_gene.innovation_num == genome2_gene.innovation_num:
                    # same innovation number, pick gene randomly from mom or dad
                    index = randint(0, 1)
                    selected_gene = (genome1_gene, genome2_gene)[index]
                    selected_genome = (genome1, genome2)[index]
                    it_genome1 += 1
                    it_genome2 += 1
                elif genome2_gene.innovation_num < genome1_gene.innovation_num:
                    # dad has lower innovation number, pick dad's gene, if they are better
                    if better_genome == genome2:
                        selected_gene = genome2.connections[it_genome2]
                        selected_genome = genome2
                    it_genome2 += 1
                elif genome1_gene.innovation_num < genome2_gene.innovation_num:
                    # mum has lower innovation number, pick mum's gene, if they are better
                    if better_genome == genome1:
                        selected_gene = genome1_gene
                        selected_genome = genome1
                    it_genome1 += 1
            elif genome1_gene is None and genome2_gene:
                # end of mum's genome, pick dad's gene, if they are better
                if better_genome == genome2:
                    selected_gene = genome2.connections[it_genome2]
                    selected_genome = genome2
                it_genome2 += 1
            elif genome1_gene and genome2_gene is None:
                # end of dad's genome, pick mum's gene, if they are better
                if better_genome == genome1:
                    selected_gene = genome1_gene
                    selected_genome = genome1
                it_genome1 += 1

            # add gene only when it has not already been added
            if selected_gene and len(child_connections) and child_connections[
                len(child_connections) - 1].innovation_num == selected_gene.innovation_num:
                print('Gene has been already added!')
                selected_gene = None

            if selected_gene is not None:
                # inherit connections
                child_connections.append(deepcopy(selected_gene))

                # inherit nodes
                if selected_gene.in_node not in node_ids:
                    node = selected_genome.exist_node(selected_gene.in_node)
                    if node is not None:
                        child_nodes.append(deepcopy(node))
                        node_ids.add(selected_gene.in_node)
                if selected_gene.out_node not in node_ids:
                    node = selected_genome.exist_node(selected_gene.out_node)
                    if node is not None:
                        child_nodes.append(deepcopy(node))
                        node_ids.add(selected_gene.out_node)

        # add in- and output neurons if they are not connected
        for node in genome1.get_bias_input_output_nodes():
            if node.node_id not in node_ids:
                child_nodes.append(deepcopy(node))
                node_ids.add(node.node_id)

        if all([con.disabled for con in child_connections]):
            choice(child_connections).disabled = False

        innovation_set = genome1.innovation_set
        inputs_num = genome1.inputs_num
        outputs_num = genome1.outputs_num
        child = Genome(child_id, innovation_set, child_nodes, child_connections, inputs_num, outputs_num)

        return child

    @staticmethod
    def compatibility_score(genome1, genome2):
        n_match, n_disjoint, n_excess = 0, 0, 0
        weight_difference = 0.0

        n_genome1 = len(genome1.connections)
        n_genome2 = len(genome2.connections)
        it_genome1, it_genome2 = 0, 0

        while it_genome1 < n_genome1 or it_genome2 < n_genome2:
            # excess
            if it_genome1 == n_genome1:
                n_excess += 1
                it_genome2 += 1
                continue

            if it_genome2 == n_genome2:
                n_excess += 1
                it_genome1 += 1
                continue

            con1 = genome1.connections[it_genome1]
            con2 = genome2.connections[it_genome2]

            # match
            if con1.innovation_num == con2.innovation_num:
                n_match += 1
                it_genome1 += 1
                it_genome2 += 1
                weight_difference += abs(con1.weight - con2.weight)
                continue

            # disjoint
            if con1.innovation_num < con2.innovation_num:
                n_disjoint += 1
                it_genome1 += 1

            if con1.innovation_num > con2.innovation_num:
                n_disjoint += 1
                it_genome2 += 1
                continue
        c1, c2, c3 = 1.0, 1.0, 0.4
        n_match += 1
        score = (c1 * n_excess + c2 * n_disjoint) / max(n_genome1, n_genome2) + c3 * weight_difference / n_match
        return score
