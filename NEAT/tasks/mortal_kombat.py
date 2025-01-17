import os
import pickle

import cv2
import numpy as np
import retro

from NEAT.src.config import ConfigMortal
from NEAT.src.genetic_algorithm import GeneticAlgorithm
from NEAT.src.network import Network
from NEAT.src.statistics import Statistics


def evaluate_genome(genomes):
    env = retro.make('MortalKombat3-Genesis', 'Level1.ShangTsungVsLiuKang.state')
    for genome_id, genome in genomes:
        ob = env.reset()

        inx, iny, inc = env.observation_space.shape
        inx = int(inx / 8)
        iny = int(iny / 8)
        network = Network.create(genome)

        done = False
        fitness_current = 0.0
        max_fitness_current = 0.0
        enemy_health = 166
        health = 166
        counter = 0

        while not done:
            env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            img_array = np.ndarray.flatten(ob)
            output = network.activate(img_array)

            ob, rew, done, info = env.step(output)

            actual_health = info['health']
            actual_enemy_health = info['enemy_health']

            if enemy_health > actual_enemy_health:
                fitness_current += enemy_health - actual_enemy_health
                enemy_health = actual_enemy_health

            if health > actual_health:
                fitness_current -= (health - actual_health)
                health = actual_health

            if actual_health <= 0:
                done = True
                fitness_current = 0

            if fitness_current > max_fitness_current:
                max_fitness_current = fitness_current
                counter = 0
                if actual_enemy_health <= 0:
                    # fitness_current = 166
                    counter = 200
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
        with open('results/genomes/mortal/winner_mortal_4_50_100_fs_neat.pkl', 'rb') as input_file:
            default_genome = pickle.load(input_file)
    except FileNotFoundError:
        print('No previous winner data! Create new genome set')
        default_genome = None

    config = ConfigMortal()
    stats = Statistics(task_name='mortal_4_300_100_fs_neat')
    population = GeneticAlgorithm(config, default_genome, stats)
    winner = population.run(evaluate_genome, 100)

    # print(f'\nBEST GENOME:\n{winner}')

    dir_name = './results/'
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    with open('results/winner_mortal_4_300_100_fs_neat.pkl', 'wb') as output:
        pickle.dump(winner, output, protocol=pickle.HIGHEST_PROTOCOL)

    # stats.draw_genome(winner)
    stats.draw_stats()
    stats.draw_species()


if __name__ == '__main__':
    run()
