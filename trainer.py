from src.kesslergame import Scenario, GraphicsType, KesslerController, TrainerEnvironment
from genetic_controller import GeneticController
import math
import numpy as np
import EasyGA
import os

class Trainer():

    def __init__(self):
        self.count = 0
        self.my_test_scenario = Scenario(
            name='Test Scenario',
            num_asteroids=10,
            ship_states=[
                {'position': (400, 400), 'angle': 90, 'lives': 3, 'team': 1, "mines_remaining": 3},
            ],
            map_size=(1000, 800),
            time_limit=60,
            ammo_limit_multiplier=0,
            stop_if_no_ammo=False
        )
        self.game_settings = {
            'perf_tracker': True,
            'graphics_type': GraphicsType.Tkinter,
            'realtime_multiplier': 1,
            'graphics_obj': None,
            'frequency': 30
        }

    def generate_chromosomes(self):
        return [
            # bullet time
            np.random.uniform(0, 0.05),
            np.random.uniform(0, 0.1),
            np.random.uniform(0.05, 0.1),
            # theta delta
            np.random.uniform(-1 * math.pi / 3, 0),
            np.random.uniform(-1 * math.pi / 3, 0),
            np.random.uniform(-1 * math.pi / 6, math.pi / 6),
            np.random.uniform(0, 1 * math.pi / 3),
            np.random.uniform(0, math.pi / 3),
            # ship turn
            np.random.uniform(-90, -30),
            np.random.uniform(-90, 0),
            np.random.uniform(-30, 30),
            np.random.uniform(0, 90),
            np.random.uniform(30, 90),
            # aster distance
            np.random.uniform(0, 125),
            np.random.uniform(0, 300),
            np.random.uniform(125, 300),
            # aster size
            np.random.uniform(0, 2),
            np.random.uniform(0, 4),
            np.random.uniform(2, 4),
            # collision time
            np.random.uniform(0, 2.5),
            np.random.uniform(0, 5),
            np.random.uniform(2.5, 5),
            # aster size
            np.random.uniform(0, 1),
            np.random.uniform(0, 2),
            np.random.uniform(1, 3),
            np.random.uniform(2, 4),
            np.random.uniform(3, 4),
            # aster distance
            np.random.uniform(0, 150),
            np.random.uniform(0, 300),
            np.random.uniform(150, 300),
        ]

    def fitness(self, chromosome):
        self.count += 1
        print(self.count)

        total_score = 0

        # game = TrainerEnvironment(settings=self.game_settings)
        # score, _ = game.run(scenario=self.my_test_scenario, controllers=[GeneticController(chromosome)])

        for _ in range(5):
            game = TrainerEnvironment(settings=self.game_settings)
            score, perf_data = game.run(scenario=self.my_test_scenario, controllers=[GeneticController(chromosome.gene_value_list[0])])
            asteroids_hit = [team.asteroids_hit for team in score.teams][0]
            total_score += asteroids_hit
        return total_score

    def get_best_chromosome(self):
        ga = EasyGA.GA()
        ga.gene_impl = lambda: self.generate_chromosomes()
        ga.chromosome_length = 1
        ga.population_size = 20
        ga.target_fitness_type = 'max'
        ga.generation_goal = 5
        ga.fitness_function_impl = self.fitness

        # try:
        ga.evolve()
        # except:
        #     # Stupid GA doesn't clean its own database after failure
        #     file_path = ['./database.db', './database.db-journal']
        #     for file in file_path:
        #         if os.path.exists(file):
        #             os.remove(file)

        best_chromosome = ga.population[0]
        print(best_chromosome)
        return best_chromosome