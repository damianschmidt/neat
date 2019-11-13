from abc import ABC, abstractmethod
from random import random, gauss, choice


class Gene(ABC):
    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def crossover(self, other_gene):
        pass

    @abstractmethod
    def mutate(self, configuration):
        pass

    @abstractmethod
    def distance(self, other_gene, configuration):
        pass


class NodeGene(Gene):
    def __init__(self, node_id):
        self.node_id = node_id
        self.bias = 0.0
        self.response = 0.0
        self.activation = 'sigmoid'
        self.aggregation = 'sum'

    def copy(self):
        node_copy = self.__class__(self.node_id)
        node_copy.bias = self.bias
        node_copy.response = self.response
        node_copy.activation = self.activation
        node_copy.aggregation = self.aggregation
        return node_copy

    def crossover(self, other_node):
        assert self.node_id == other_node.node_id
        crossover_node = self.__class__(self.node_id)
        crossover_node.bias = self.bias if random() > 0.5 else other_node.bias
        crossover_node.response = self.response if random() > 0.5 else other_node.response
        crossover_node.activation = self.activation if random() > 0.5 else other_node.activation
        crossover_node.aggregation = self.aggregation if random() > 0.5 else other_node.aggregation
        return crossover_node

    def mutate(self, config):
        self.bias = self.mutate_bias(config)
        self.response = self.mutate_response(config)
        self.activation = self.mutate_activation(config)
        self.aggregation = self.mutate_aggregation(config)

    def mutate_bias(self, config):
        mutate_rate = config.bias_mutate_rate
        replace_rate = config.bias_replace_rate
        min_value = config.bias_min_value
        max_value = config.bias_max_value
        rand = random()

        if rand < mutate_rate:
            mutate_power = config.bias_mutate_power
            return max(min(self.bias + gauss(0.0, mutate_power), max_value), min_value)

        if rand < replace_rate + mutate_rate:
            mean = config.bias_init_mean
            stdev = config.bias_init_stdev
            value = gauss(mean, stdev)
            return max(min(value, max_value), min_value)
        return self.bias

    def mutate_response(self, config):
        mutate_rate = config.response_mutate_rate
        replace_rate = config.response_replace_rate
        min_value = config.response_min_value
        max_value = config.response_max_value
        rand = random()

        if rand < mutate_rate:
            mutate_power = config.response_mutate_power
            return max(min(self.response + gauss(0.0, mutate_power), max_value), min_value)

        if rand < replace_rate + mutate_rate:
            mean = config.response_init_mean
            stdev = config.response_init_stdev
            value = gauss(mean, stdev)
            return max(min(value, max_value), min_value)
        return self.response

    def mutate_activation(self, config):
        mutate_rate = config.activation_mutate_rate

        if mutate_rate > 0:
            if random() < mutate_rate:
                options = config.activation_options
                return choice(options)
        return self.activation

    def mutate_aggregation(self, config):
        mutate_rate = config.aggregation_mutate_rate

        if mutate_rate > 0:
            if random() < mutate_rate:
                options = config.aggregation_options
                return choice(options)
        return self.aggregation

    def distance(self, other_node, config):
        distance = abs(self.bias - other_node.bias) + abs(self.response - other_node.response)
        if self.activation != other_node.activation:
            distance += 1.0
        if self.aggregation != other_node.aggregation:
            distance += 1.0
        return distance * config.compatibility_weight_coefficient

    def __str__(self):
        string = f'Node ID: {self.node_id}, Bias: {self.bias}, Response: {self.response},' \
                 f' Activation: {self.activation}, Aggregation: {self.aggregation}'
        return string


class ConnectionGene(Gene):
    def __init__(self, connection_id):
        self.connection_id = connection_id
        self.weight = 0.0
        self.enabled = True

    def copy(self):
        connection_copy = self.__class__(self.connection_id)
        connection_copy.weight = self.weight
        connection_copy.enabled = self.enabled
        return connection_copy

    def crossover(self, other_connection):
        assert self.connection_id == other_connection.connection_id
        crossover_connection = self.__class__(self.connection_id)
        crossover_connection.weight = self.weight if random() > 0.5 else other_connection.weight
        crossover_connection.enabled = self.enabled if random() > 0.5 else other_connection.enabled
        return crossover_connection

    def mutate(self, config):
        self.weight = self.mutate_weight(config)
        self.enabled = self.mutate_enabled(config)

    def mutate_weight(self, config):
        mutate_rate = config.weight_mutate_rate
        replace_rate = config.weight_replace_rate
        min_value = config.weight_min_value
        max_value = config.weight_max_value
        rand = random()

        if rand < mutate_rate:
            mutate_power = config.weight_mutate_power
            return max(min(self.weight + gauss(0.0, mutate_power), max_value), min_value)

        if rand < replace_rate + mutate_rate:
            mean = config.weight_init_mean
            stdev = config.weight_init_stdev
            value = gauss(mean, stdev)
            return max(min(value, max_value), min_value)
        return self.weight

    def mutate_enabled(self, config):
        mutate_rate = config.enabled_mutate_rate
        if mutate_rate > 0:
            if random() < mutate_rate:
                return random() < 0.5
        return self.enabled

    def distance(self, other_connection, config):
        distance = abs(self.weight - other_connection.weight)
        if self.enabled != other_connection.enabled:
            distance += 1.0
        return distance * config.compatibility_weight_coefficient

    def __str__(self):
        string = f'Connection ID: {self.connection_id}, Weight: {self.weight}, Enabled: {self.enabled}'
        return string
