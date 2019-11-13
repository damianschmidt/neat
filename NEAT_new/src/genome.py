class Genome:
    def __init__(self, genome_id):
        self.genome_id = genome_id
        self.connections = {}
        self.nodes = {}
        self.fitness = None

    def create_new(self):
        pass

    def crossover(self, genome1, genome2):
        assert isinstance(genome1.fitness, (int, float))
        assert isinstance(genome2.fitness, (int, float))

        # choose more fit parent
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent2, parent1 = genome1, genome2

        # inherit nodes
        parent1_nodes = parent1.nodes
        parent2_nodes = parent2.nodes

        for gene_id, node_gene1 in parent1_nodes.items():
            node_gene2 = parent2_nodes.get(gene_id)
            assert gene_id not in self.nodes
            if node_gene2 is None:
                # more gene in fittest parent - get it
                self.nodes[gene_id] = node_gene1.copy()
            else:
                # combine from both
                self.nodes[gene_id] = node_gene1.crossover(node_gene2)

        # inherit connections
        for connection_id, connection_gene1 in parent1.connections.items():
            connection_gene2 = parent2.connections.get(connection_id)
            if connection_gene2 is None:
                # excess or disjoint - copy connection from parent1
                self.connections[connection_id] = connection_gene1.copy()
            else:
                # combine from both
                self.connections[connection_id] = connection_gene1.crossover(connection_gene2)

    def mutate(self):
        pass

    def distance(self):
        pass

    def size(self):
        number_of_enabled_conn = sum([1 for connection_gene in self.connections.values() if connection_gene.enabled])
        return len(self.nodes), number_of_enabled_conn

    def __str__(self):
        string = f'ID: {self.genome_id}\nFitness: {self.fitness}\nNodes:'
        for k, node_gene in self.nodes.items():
            string += f'\n\t{k} {node_gene}'
        string += '\nConnections:'
        connections = list(self.connections.values())
        connections.sort()
        for conn in connections:
            string += f'\n\t {conn}'
        return string
