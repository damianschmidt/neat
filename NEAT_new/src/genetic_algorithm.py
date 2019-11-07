import os
import shutil
import time
from copy import deepcopy
from random import randint, choice

from NEAT_new.src.genome import Genome
from NEAT_new.src.innovation import InnovationSet
from NEAT_new.src.species import Species


class GeneticAlgorithm:
    def __init__(self, task):
        self.genomes = []
        self.species = []
        self.bests = []
        self.best = None
        self.best_previous = None
        self.innovation_set = InnovationSet()
        self.task = task
        self.generation = 0

        self.next_genome_id = 0
        self.next_species_id = 0

        self.init_time = time.time()
        self.last_time = time.time()

        # parameters
        self.population_size = 100
        self.number_generation_allowed_to_not_improve = 20
        self.crossover_rate = 0.7
        self.survival_rate = 0.3
        self.compatibility_threshold = 3.0
        self.target_species = 30

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
        self.clear_species()
        # reproduction
        self.reproduction()
        # create basic population / initial birth
        self.create_basic_population()
        # create new phenotypes and evaluate network
        for g in self.genomes:
            network = g.create_pheontype()  # TODO: has to be done
            fitness, solved = self.task.evaluate(network)
            g.fitness = fitness
            g.solved = int(solved)
        # sort genomes reverse by fitness
        self.sort_genomes_reverse_by_fitness()
        # update best
        self.update_best()
        # assign genomes to species
        self.assign_genomes_to_species()
        # remove empty species
        self.remove_empty_species()
        # adjust compatibility_threshold
        self.adjust_compatibility_threshold()
        # sort members by fitness, adjust fitness, average_fitness and max_fitness
        self.update_species()
        # update statistics
        self.update_statistics()
        # print representation
        print(self)

        # TODO: visualize

    def update_species(self):
        for s in self.species:
            # sort members of species reverse by fitness
            s.members.sort(key=lambda x: x.fitness, reverse=True)
            s.leader_old = s.leader
            s.leader = s.members[0]

            if s.leader.fitness > s.max_fitness:
                s.generations_not_improved = 0
            else:
                s.generations_not_improved += 1
            s.max_fitness = s.leader.fitness

            # should I boost young species and punish old?
            self.boost_punish_species(s)

    @staticmethod
    def boost_punish_species(s):
        sum_fitness = 0.0
        for member in s.members:
            fitness = member.fitness
            sum_fitness += fitness
            # bonus to young species
            if s.age < 10:
                fitness *= 1.3
            # punish for old species
            elif s.age > 50:
                fitness *= 0.7
            member.adjusted_fitness = fitness / len(s.members)
        s.average_fitness = sum_fitness / len(s.members)

    def update_statistics(self):
        for s in self.species:
            while len(self.statistics['species']) <= s.species_id:
                self.statistics['species'].append([])
            stat_data = [self.generation, len(s.members), s.leader.fitness, s.leader.solved]
            self.statistics['species'][s.species_id].append(stat_data)

    def adjust_compatibility_threshold(self):
        if len(self.species) < self.target_species:
            self.compatibility_threshold *= 0.95
        elif len(self.species) > self.target_species:
            self.compatibility_threshold *= 1.05

    def remove_empty_species(self):
        self.species[:] = filter(lambda s: len(s.members) > 0, self.species)

    def assign_genomes_to_species(self):
        for g in self.genomes:
            added_to_species = False
            for s in self.species:
                # add to existing if fit
                compatibility_score = self.compatibility_score(g, s.leader)
                if compatibility_score <= self.compatibility_threshold:
                    s.add_member(g)
                    added_to_species = True
                    break

            if not added_to_species:
                s = Species(g, self.next_species_id)
                self.next_species_id += 1
                self.species.append(s)

    def update_best(self):
        self.best_previous = self.best
        if self.best is None or self.best.fitness < self.genomes[0].fitness:
            self.best = self.genomes[0]

    def sort_genomes_reverse_by_fitness(self):
        self.genomes.sort(key=lambda x: x.fitness, reverse=True)

    def create_basic_population(self):
        while len(self.genomes) < self.population_size:
            genome = Genome(self.next_genome_id, self.innovation_set, None, None, self.task.inputs_num,
                            self.task.outputs_num)
            self.genomes.append(genome)
            self.next_genome_id += 1

    def clear_species(self):
        species = []
        for s in self.species:
            if s.generations_not_improved < self.number_generation_allowed_to_not_improve and s.spawns_required > 0:
                species.append(s)
        self.species[:] = species

    def reproduction(self):
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

    def __str__(self):
        best = self.best
        species_ids = ' '.join(s.species_id for s in self.species)
        species_members_length = [len(s.members) for s in self.species]
        species_age = ' '.join(s.age for s in self.species)
        species_not_improved = ' '.join(s.generations_not_improved for s in self.species)
        species_max_fitness = ' '.join(s.max_fitness for s in self.species)
        species_avg_fitness = ' '.join(s.average_fitness for s in self.species)
        species_leader = ' '.join(s.leader for s in self.species)
        species_solved = ' '.join(s.leader.solved for s in self.species)
        now = time.time()
        string = f'\nGeneration: {self.generation}' \
                 f'\nBest ID: {best.genome_id}, Fitness: {best.fitness}, Nodes: {len(best.nodes)},' \
                 f' Connections: {len(best.connections)}, Depth: {best.phenotype.depth}' \
                 f'\nSpecies ID: {species_ids}' \
                 f'\nMembers length: {species_members_length}' \
                 f'\nAge: {species_age}' \
                 f'\nNot improved: {species_not_improved}' \
                 f'\nMax fitness: {species_max_fitness}' \
                 f'\nAverage fitness: {species_avg_fitness}' \
                 f'\nLeaders: {species_leader}' \
                 f'\nSolved: {species_solved}' \
                 f'\nPopulation length: {len(self.genomes)}, Species length: {len(self.species)}, ' \
                 f'Compatibility threshold: {self.compatibility_threshold}' \
                 f'\nTotal time: {now-self.init_time}, Time per generation: {now-self.last_time}\n'
        self.last_time = now
        return string
