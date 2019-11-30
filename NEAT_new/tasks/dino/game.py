import os
import pickle

import pygame
from random import randint
from math import floor

from NEAT_new.src.config import ConfigDino
from NEAT_new.src.genetic_algorithm import GeneticAlgorithm
from NEAT_new.src.network import Network
from NEAT_new.src.statistics import Statistics
from NEAT_new.tasks.dino.dino import Dino
from NEAT_new.tasks.dino.ground import Ground
from NEAT_new.tasks.dino.cactus import Cactus
from NEAT_new.tasks.dino.bird import Bird

pygame.init()
pygame.display.set_caption('Dino')
clock = pygame.time.Clock()


class Game:
    def __init__(self):
        self.screen_width = 1000
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.score = 0
        self.speed = 8
        self.run = False
        self.generation = 0

    def move(self, dinos, ground, obstacles):
        [dino.move() for dino in dinos]
        for obstacle in obstacles:
            if obstacle.x < self.screen_width and not obstacle.next_added:
                obstacle.next_added = True
                random = randint(0, 3)
                obstacles.append(Cactus(self.screen_width + randint(400, 1000), self.speed)) if random != 3 \
                    else obstacles.append(Bird(self.screen_width + randint(400, 1000), self.speed))
        [obstacle.move() for obstacle in obstacles]
        [obstacles.remove(obstacle) for obstacle in obstacles if obstacle.x + obstacle.img.get_width() < 0]
        ground.move()

    def draw_score(self, dinos):
        pygame.font.init()
        font = pygame.font.SysFont('Comic-Sans', 20)
        score_surface = font.render(f'Score: {floor(self.score)}', True, (0, 0, 0))
        alive_surface = font.render(f'Alive: {floor(len(dinos))}', True, (0, 0, 0))
        generation_surface = font.render(f'Generation: {floor(self.generation)}', True, (0, 0, 0))
        self.screen.blit(score_surface, (10, 10))
        self.screen.blit(alive_surface, (10, 30))
        self.screen.blit(generation_surface, (10, 50))

    def draw_screen(self, dinos, ground, obstacles):
        self.screen.fill((255, 255, 255))

        ground.draw(self.screen)
        [obstacle.draw(self.screen, ground) for obstacle in obstacles]
        [dino.draw(self.screen, ground) for dino in dinos]

        self.draw_score(dinos)

    def collide(self, dinos, obstacles):
        for dino in dinos:
            for obstacle in obstacles:
                if obstacle.collide(dino):
                    self.run = False

    def restart(self):
        self.speed = 8.0
        self.score = 0
        self.run = True
        return [Dino()], Ground(self.screen_width), [Cactus(self.screen_width, self.speed)]

    def increase_speed(self, ground, obstacles, value):
        self.speed += value
        ground.velocity = self.speed
        for obstacle in obstacles:
            obstacle.velocity = self.speed

    def game_loop(self):
        dinos = [Dino()]
        ground = Ground(self.screen_width)
        obstacles = [Cactus(self.screen_width, self.speed)]
        self.run = True

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    dinos, ground, obstacles = self.restart()

            while self.run:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        quit()
                    elif event.type == pygame.KEYDOWN and (event.key == pygame.K_SPACE or event.key == pygame.K_UP):
                        dinos[0].jump()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                        dinos[0].duck()
                    elif event.type == pygame.KEYUP and (
                            event.key == pygame.K_SPACE or event.key == pygame.K_UP or event.key == pygame.K_DOWN):
                        dinos[0].run()

                self.move(dinos, ground, obstacles)
                if int(self.score % 15) == 0:
                    self.increase_speed(ground, obstacles, 0.1)
                self.draw_screen(dinos, ground, obstacles)
                self.collide(dinos, obstacles)
                self.score += 0.2

                clock.tick(30)
                pygame.display.update()

    def eval_genomes(self, genomes):
        self.generation += 1

        nets = []
        dinos = []
        genomes_list = []

        for genome_id, genome in genomes:
            genome.fitness = 0
            net = Network.create(genome)

            nets.append(net)
            dinos.append(Dino())
            genomes_list.append(genome)

        # Build game
        self.speed = 8
        self.score = 0
        ground = Ground(self.screen_width)
        obstacles = [Cactus(self.screen_width, self.speed)]
        tick = 0

        while len(dinos) > 0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            # On which pipe NEAT should take care of
            obstacle_index = 0
            if len(obstacles) > 1 and dinos[0].x > obstacles[0].x + obstacles[0].img.get_width():
                obstacle_index = 1

            for i, dino in enumerate(dinos):
                genomes_list[i].fitness += 0.1
                dino.move()

                distance_to_next_obstacle = obstacles[obstacle_index].x - (dino.x + dino.img.get_width())
                if type(obstacles[obstacle_index]) == Cactus:
                    height_of_obstacle = obstacles[obstacle_index].img.get_height()
                else:
                    height_of_obstacle = obstacles[obstacle_index].img.get_height() + obstacles[obstacle_index].y_type
                width_of_obstacle = obstacles[obstacle_index].img.get_width()
                speed = self.speed
                dino_y = dino.y_change
                if len(obstacles) > obstacle_index + 1:
                    gap_between_obstacles = obstacles[obstacle_index + 1].x - obstacles[obstacle_index].x
                else:
                    gap_between_obstacles = self.screen_width

                output = nets[i].activate(
                    (distance_to_next_obstacle, height_of_obstacle, width_of_obstacle, speed, dino_y,
                     gap_between_obstacles))

                if i == 0:
                    self.draw_screen(dinos, ground, obstacles)
                    self.draw_what_dino_know(distance_to_next_obstacle, height_of_obstacle, width_of_obstacle, speed,
                                             dino_y, gap_between_obstacles)
                    clock.tick(30)
                    pygame.display.update()

                decision = [output[0], 0] if output[0] > output[1] else [output[1], 1]
                if decision[0] > 0.5:
                    dino.jump() if decision[1] == 0 else dino.duck()
                else:
                    dino.run()

            ground.move()
            for obstacle in obstacles:
                if obstacle.x < self.screen_width and not obstacle.next_added:
                    obstacle.next_added = True
                    random = randint(0, 3)
                    obstacles.append(Cactus(self.screen_width + randint(400, 1000), self.speed)) if random != 3 \
                        else obstacles.append(Bird(self.screen_width + randint(400, 1000), self.speed))
            [obstacle.move() for obstacle in obstacles]
            [obstacles.remove(obstacle) for obstacle in obstacles if obstacle.x + obstacle.img.get_width() < 0]

            for obstacle in obstacles:
                for i, dino in enumerate(dinos):

                    if not obstacle.passed and obstacle.x < dino.x:
                        genomes_list[i].fitness += 5
                        obstacle.passed = True

            to_remove = []
            for i, dino in enumerate(dinos):
                for obstacle in obstacles:
                    if obstacle.collide(dino):
                        to_remove.append((dino, nets[i], genomes_list[i]))

            for item in to_remove:
                dinos.remove(item[0])
                nets.remove(item[1])
                genomes_list.remove(item[2])

            if tick % 5 == 0:
                tick = 0
                self.score += 1
                if self.score % 30 == 0:
                    self.increase_speed(ground, obstacles, 1)

            tick += 1

    def run_neat(self):
        try:
            with open('./results/winner_dino_fs_neat.pkl', 'rb') as input_file:
                default_genome = pickle.load(input_file)
        except FileNotFoundError:
            print('No previous winner data! Create new genome set')
            default_genome = None

        config = ConfigDino()
        stats = Statistics(task_name='dino_fs_neat')
        p = GeneticAlgorithm(config, default_genome, stats)
        winner = p.run(self.eval_genomes, 50)
        print(f'\nBEST GENOME:\n{winner}')

        dir_name = './results/'
        os.makedirs(os.path.dirname(dir_name), exist_ok=True)
        with open('results/winner_dino_fs_neat.pkl', 'wb') as output:
            pickle.dump(winner, output, protocol=pickle.HIGHEST_PROTOCOL)

        stats.draw_genome(winner)
        stats.draw_stats()
        stats.draw_species()

    def draw_what_dino_know(self, distance_to_next_obstacle, height_of_obstacle, width_of_obstacle, speed,
                            dino_y, gap_between_obstacles):
        pygame.font.init()
        font = pygame.font.SysFont('Comic-Sans', 20)
        distance_to_next_obstacle_surface = font.render(f'Distance: {floor(distance_to_next_obstacle)}', True,
                                                        (255, 0, 0))
        height_of_obstacle_surface = font.render(f'Height: {floor(height_of_obstacle)}', True, (255, 0, 0))
        width_of_obstacle_surface = font.render(f'Width: {floor(width_of_obstacle)}', True, (255, 0, 0))
        speed_surface = font.render(f'Speed: {floor(speed)}', True, (255, 0, 0))
        dino_y_surface = font.render(f'Dino Y: {floor(dino_y)}', True, (255, 0, 0))
        gap_between_obstacles_surface = font.render(f'Gap: {floor(gap_between_obstacles)}', True, (255, 0, 0))

        self.screen.blit(distance_to_next_obstacle_surface, (500, 10))
        self.screen.blit(height_of_obstacle_surface, (500, 30))
        self.screen.blit(width_of_obstacle_surface, (500, 50))
        self.screen.blit(speed_surface, (500, 70))
        self.screen.blit(dino_y_surface, (500, 90))
        self.screen.blit(gap_between_obstacles_surface, (500, 110))
