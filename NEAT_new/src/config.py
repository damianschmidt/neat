class Config:
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
    node_delete_prob = 0.2
    conn_add_prob = 0.5
    conn_delete_prob = 0.5

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
