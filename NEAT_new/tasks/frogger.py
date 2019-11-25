import os
import pickle

import cv2
import numpy as np
import retro

from NEAT_new.src.config import ConfigFrogger
from NEAT_new.src.genetic_algorithm import GeneticAlgorithm
from NEAT_new.src.network import Network
from NEAT_new.src.statistics import Statistics


def evaluate_genome(genomes):
    env = retro.make('Frogger-Genesis', '1Player.Level1.state')
    for genome_id, genome in genomes:
        ob = env.reset()

        inx, iny, inc = env.observation_space.shape
        inx = int(inx / 8)
        iny = int(iny / 8)
        network = Network.create(genome)

        done = False

        while not done:
            env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            img_array = np.ndarray.flatten(ob)
            output = network.activate(img_array)

            ob, rew, done, info = env.step(output)

            # add rewarding

    env.render(close=True)


def run():
    try:
        with open('./results/winner_frogger.pkl', 'rb') as input_file:
            default_genome = pickle.load(input_file)
    except FileNotFoundError:
        print('No previous winner data! Create new genome set')
        default_genome = None

    config = ConfigFrogger()
    stats = Statistics()
    population = GeneticAlgorithm(config, default_genome, stats)
    winner = population.run(evaluate_genome, 50)

    print(f'\nBEST GENOME:\n{winner}')

    dir_name = './results/'
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)
    with open('results/winner_frogger.pkl', 'wb') as output:
        pickle.dump(winner, output, protocol=pickle.HIGHEST_PROTOCOL)

    stats.draw_genome(winner)
    stats.draw_stats()
    stats.draw_species()


if __name__ == '__main__':
    run()
