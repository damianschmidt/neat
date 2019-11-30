import os
import pickle

import cv2
import numpy as np
import retro

from NEAT_new.src.config import ConfigSonic
from NEAT_new.src.genetic_algorithm import GeneticAlgorithm
from NEAT_new.src.network import Network
from NEAT_new.src.statistics import Statistics


def evaluate_genome(genomes):
    env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
    for genome_id, genome in genomes:
        ob = env.reset()

        inx, iny, inc = env.observation_space.shape
        inx = int(inx / 8)
        iny = int(iny / 8)
        network = Network.create(genome)

        done = False
        fitness_current = 0
        current_max_fitness = 0
        counter = 0
        xpos_max = 0
        rings_max = 0

        while not done:
            env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            img_array = np.ndarray.flatten(ob)
            output = network.activate(img_array)

            ob, rew, done, info = env.step(output)

            xpos = info['x']
            xpos_end = info['screen_x_end']
            rings = info['rings']

            fitness_current += rew

            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos

            if rings > rings_max:
                fitness_current += 10
                rings_max = rings

            if xpos == xpos_end and xpos > 100:
                fitness_current = xpos_end

            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1

            if counter == 300:
                done = True

            if done:
                print(genome_id, genome.fitness)

            genome.fitness = fitness_current
    env.render(close=True)


def run():
    try:
        with open('results/genomes/sonic/winner_sonic_4_50_150_fs_neat.pkl', 'rb') as input_file:
            default_genome = pickle.load(input_file)
    except FileNotFoundError:
        print('No previous winner data! Create new genome set')
        default_genome = None

    config = ConfigSonic()
    stats = Statistics(task_name='sonic_4_50_150_fs_neat_extra_rew')
    population = GeneticAlgorithm(config, default_genome, stats)
    winner = population.run(evaluate_genome, 100)

    # print(f'\nBEST GENOME:\n{winner}')

    dir_name = './results/'
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    with open('results/genomes/sonic/winner_sonic_4_50_150_fs_neat_extra_rew.pkl', 'wb') as output:
        pickle.dump(winner, output, protocol=pickle.HIGHEST_PROTOCOL)

    # stats.draw_genome(winner)
    stats.draw_stats()
    stats.draw_species()


if __name__ == '__main__':
    run()
