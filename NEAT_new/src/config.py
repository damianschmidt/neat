class Config:
    # NEAT parameters
    population_size = 150
    fitness_threshold = 3.9

    # genome default structure
    num_inputs = 2
    num_hidden = 0
    num_outputs = 1

    initial_connection = 'full'

    # genome compatibility
    compatibility_weight_coefficient = 0.5
    compatibility_disjoint_coefficient = 1.0

    # genome mutation
    node_add_prob = 0.2
    node_remove_prob = 0.2
    conn_add_prob = 0.5
    conn_remove_prob = 0.5

    # node mutation
    bias_mutate_rate = 0.7
    bias_mutate_power = 0.5
    bias_replace_rate = 0.1
    bias_max_value = 30.0
    bias_min_value = -30.0
    bias_init_mean = 0.0
    bias_init_stdev = 1.0

    response_mutate_rate = 0.0
    response_mutate_power = 0.0
    response_replace_rate = 0.0
    response_max_value = 30.0
    response_min_value = -30.0
    response_init_mean = 1.0
    response_init_stdev = 0.0

    activation_mutate_rate = 0.0
    activation_options = ['sigmoid']

    aggregation_mutate_rate = 0.0
    aggregation_options = ['sum']

    # connection mutation
    weight_mutate_rate = 0.8
    weight_mutate_power = 0.5
    weight_replace_rate = 0.1
    weight_max_value = 30.0
    weight_min_value = -30.0
    weight_init_mean = 0.0
    weight_init_stdev = 1.0

    enabled_mutate_rate = 0.01

    # species
    compatibility_threshold = 3.0

    # reproduction
    elitism = 2
    survival_threshold = 0.2

    # stagnation
    species_fitness_func = max
    max_stagnation = 20


class ConfigFlappyBird(Config):
    # NEAT parameters
    fitness_threshold = 5000

    # genome default structure
    num_inputs = 3
    num_hidden = 0
    num_outputs = 1


class ConfigDino(Config):
    # NEAT parameters
    fitness_threshold = 650

    # genome default structure
    num_inputs = 6
    num_hidden = 0
    num_outputs = 2


class ConfigSonic(Config):
    # NEAT parameters
    population_size = 20
    fitness_threshold = 100000

    # genome default structure
    num_inputs = 1120
    num_hidden = 0
    num_outputs = 12


class ConfigMortal(Config):
    # NEAT parameters
    population_size = 20
    fitness_threshold = 100000

    # genome default structure
    num_inputs = 1120
    num_hidden = 0
    num_outputs = 12


class ConfigFrogger(Config):
    # NEAT parameters
    population_size = 20
    fitness_threshold = 100000

    # genome default structure
    num_inputs = 1120
    num_hidden = 0
    num_outputs = 12


class ConfigF1(Config):
    # NEAT parameters
    population_size = 20
    fitness_threshold = 100000

    # genome default structure
    num_inputs = 768
    num_hidden = 0
    num_outputs = 12
