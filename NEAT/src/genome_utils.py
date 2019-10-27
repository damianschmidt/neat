def compatibility_distance(genome1, genome2, c1, c2, c3):
    excess_genes = count_excess_genes(genome1, genome2)
    disjoint_genes = count_disjoint_genes(genome1, genome2)
    avg_weight_diff = average_weight_diff(genome1, genome2)

    return excess_genes * c1 + disjoint_genes * c2 + avg_weight_diff * c3


def count_matching_genes(genome1, genome2):
    matching_genes = 0

    highest_innovation1, highest_innovation2, indices = get_nodes_innovations_and_indices(genome1, genome2)
    for i in range(indices + 1):
        node1, node2 = get_nodes(genome1, genome2, i)
        if node1 and node2:
            matching_genes += 1

    return matching_genes


def count_disjoint_genes(genome1, genome2):
    disjoint_genes = 0

    highest_innovation1, highest_innovation2, indices = get_nodes_innovations_and_indices(genome1, genome2)
    for i in range(indices + 1):
        node1, node2 = get_nodes(genome1, genome2, i)
        if node1 is None and highest_innovation1 > i and node2 is not None:
            # genome 1 lacks gene, genome 2 has gene, genome 1 has more genes with higher innovation numbers
            disjoint_genes += 1
        elif node2 is None and highest_innovation2 > i and node1 is not None:
            disjoint_genes += 1
    highest_innovation1, highest_innovation2, indices = get_connections_innovations_and_indices(genome1, genome2)
    for i in range(indices + 1):
        connection1, connection2 = get_connections(genome1, genome2, i)
        if connection1 is None and highest_innovation1 > i and connection2 is not None:
            disjoint_genes += 1
        elif connection2 is None and highest_innovation2 > i and connection1 is not None:
            disjoint_genes += 1

    return disjoint_genes


def count_excess_genes(genome1, genome2):
    excess_genes = 0

    highest_innovation1, highest_innovation2, indices = get_nodes_innovations_and_indices(genome1, genome2)
    for i in range(indices + 1):
        node1, node2 = get_nodes(genome1, genome2, i)
        if node1 is None and highest_innovation1 < i and node2 is not None:
            excess_genes += 1
        elif node2 is None and highest_innovation2 < i and node1 is not None:
            excess_genes += 1

    highest_innovation1, highest_innovation2, indices = get_connections_innovations_and_indices(genome1, genome2)
    for i in range(indices + 1):
        connection1, connection2 = get_connections(genome1, genome2, i)
        if connection1 is None and highest_innovation1 < i and connection2 is not None:
            excess_genes += 1
        elif connection2 is None and highest_innovation2 < i and connection1 is not None:
            excess_genes += 1

    return excess_genes


def get_nodes_innovations_and_indices(genome1, genome2):
    list_of_node_keys1 = sorted([*genome1.dict_of_nodes])
    list_of_node_keys2 = sorted([*genome2.dict_of_nodes])
    highest_innovation1 = list_of_node_keys1[-1]
    highest_innovation2 = list_of_node_keys2[-1]
    indices = max(highest_innovation1, highest_innovation2)
    return highest_innovation1, highest_innovation2, indices


def get_connections_innovations_and_indices(genome1, genome2):
    list_of_connection_keys1 = [*genome1.dict_of_connections]
    list_of_connection_keys1.sort()
    list_of_connection_keys2 = [*genome2.dict_of_connections]
    list_of_connection_keys2.sort()
    highest_innovation1 = list_of_connection_keys1[-1]
    highest_innovation2 = list_of_connection_keys2[-1]
    indices = max(highest_innovation1, highest_innovation2)
    return highest_innovation1, highest_innovation2, indices


def get_nodes(genome1, genome2, i):
    try:
        node1 = genome1.dict_of_nodes[i]
    except KeyError:
        node1 = None
    try:
        node2 = genome2.dict_of_nodes[i]
    except KeyError:
        node2 = None
    return node1, node2


def get_connections(genome1, genome2, i):
    try:
        connection1 = genome1.dict_of_connections[i]
    except KeyError:
        connection1 = None
    try:
        connection2 = genome2.dict_of_connections[i]
    except KeyError:
        connection2 = None
    return connection1, connection2


def average_weight_diff(genome1, genome2):
    matching_genes = 0
    weight_diff = 0.0

    highest_innovation1, highest_innovation2, indices = get_connections_innovations_and_indices(genome1, genome2)
    for i in range(indices + 1):
        connection1, connection2 = get_connections(genome1, genome2, i)
        if connection1 and connection2:
            matching_genes += 1
            weight_diff += abs(connection1.weight - connection2.weight)
    return round(weight_diff, 5) / matching_genes
