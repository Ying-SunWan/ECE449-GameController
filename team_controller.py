from src.kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

BULLET_SPEED = 800

class TeamController(KesslerController):

    def __init__(self):
        self.eval_frames = 0  # What is this?
        self.setup_target_fuzzy_system()
        self.setup_move_fuzzy_system()
        self.setup_danger_fuzzy_system()

    def setup_target_fuzzy_system(self):
        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), 'bullet_time')
        theta_delta = ctrl.Antecedent(np.arange(-1 * math.pi, math.pi, 0.1), 'theta_delta')  # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')  # Degrees due to Kessler
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), 'ship_fire')

        bullet_time['S'] = fuzz.trimf(bullet_time.universe, [0, 0, 0.05])
        bullet_time['M'] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
        bullet_time['L'] = fuzz.smf(bullet_time.universe, 0.0, 0.1)

        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1 * math.pi / 3, -1 * math.pi / 6)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1 * math.pi / 3, -1 * math.pi / 6, 0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1 * math.pi / 6, 0, math.pi / 6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0, math.pi / 6, math.pi / 3])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, math.pi / 6, math.pi / 3)

        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -30])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90, -30, 0])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-30, 0, 30])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0, 30, 90])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [30, 180, 180])

        ship_fire['N'] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire['Y'] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        target_rules = [
            ctrl.Rule(bullet_time['L'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['L'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['Y'])),
        ]
        self.targeting_control = ctrl.ControlSystem(target_rules)

    def setup_move_fuzzy_system(self):
        aster_distance = ctrl.Antecedent(np.arange(0, 300, 0.1), 'aster_distance')
        theta_delta = ctrl.Antecedent(np.arange(-1 * math.pi, math.pi, 0.1), 'theta_delta')  # Radians due to Python
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), 'ship_turn')  # Degrees due to Kessler
        ship_thrust_rate = ctrl.Consequent(np.arange(0, 300, 0.1), 'ship_thrust_rate')

        aster_distance['S'] = fuzz.trimf(aster_distance.universe, [0, 0, 150])
        aster_distance['M'] = fuzz.trimf(aster_distance.universe, [100, 150, 200])
        aster_distance['L'] = fuzz.trimf(aster_distance.universe, [150, 300, 300])

        theta_delta['NL'] = fuzz.zmf(theta_delta.universe, -1 * math.pi / 3, -1 * math.pi / 6)
        theta_delta['NS'] = fuzz.trimf(theta_delta.universe, [-1 * math.pi / 3, -1 * math.pi / 6, 0])
        theta_delta['Z'] = fuzz.trimf(theta_delta.universe, [-1 * math.pi / 6, 0, math.pi / 6])
        theta_delta['PS'] = fuzz.trimf(theta_delta.universe, [0, math.pi / 6, math.pi / 3])
        theta_delta['PL'] = fuzz.smf(theta_delta.universe, math.pi / 6, math.pi / 3)

        ship_turn['NL'] = fuzz.trimf(ship_turn.universe, [-180, -180, -30])
        ship_turn['NS'] = fuzz.trimf(ship_turn.universe, [-90, -30, 0])
        ship_turn['Z'] = fuzz.trimf(ship_turn.universe, [-30, 0, 30])
        ship_turn['PS'] = fuzz.trimf(ship_turn.universe, [0, 30, 90])
        ship_turn['PL'] = fuzz.trimf(ship_turn.universe, [30, 180, 180])

        ship_thrust_rate['S'] = fuzz.trimf(ship_thrust_rate.universe, [0, 0, 100])
        ship_thrust_rate['MS'] = fuzz.trimf(ship_thrust_rate.universe, [50, 100, 150])
        ship_thrust_rate['M'] = fuzz.trimf(ship_thrust_rate.universe, [100, 150, 200])
        ship_thrust_rate['ML'] = fuzz.trimf(ship_thrust_rate.universe, [150, 200, 250])
        ship_thrust_rate['L'] = fuzz.trimf(ship_thrust_rate.universe, [200, 300, 300])

        moving_rules = [
            ctrl.Rule(aster_distance['S'] & theta_delta['NL'], (ship_turn['NL'], ship_thrust_rate['ML'])),
            ctrl.Rule(aster_distance['S'] & theta_delta['NS'], (ship_turn['NS'], ship_thrust_rate['ML'])),
            ctrl.Rule(aster_distance['S'] & theta_delta['Z'], (ship_turn['Z'], ship_thrust_rate['L'])),
            ctrl.Rule(aster_distance['S'] & theta_delta['PS'], (ship_turn['PS'], ship_thrust_rate['ML'])),
            ctrl.Rule(aster_distance['S'] & theta_delta['PL'], (ship_turn['PL'], ship_thrust_rate['ML'])),
            ctrl.Rule(aster_distance['M'] & theta_delta['NL'], (ship_turn['NL'], ship_thrust_rate['MS'])),
            ctrl.Rule(aster_distance['M'] & theta_delta['NS'], (ship_turn['NS'], ship_thrust_rate['M'])),
            ctrl.Rule(aster_distance['M'] & theta_delta['Z'], (ship_turn['Z'], ship_thrust_rate['M'])),
            ctrl.Rule(aster_distance['M'] & theta_delta['PS'], (ship_turn['PS'], ship_thrust_rate['M'])),
            ctrl.Rule(aster_distance['M'] & theta_delta['PL'], (ship_turn['PL'], ship_thrust_rate['MS'])),
            ctrl.Rule(aster_distance['L'] & theta_delta['NL'], (ship_turn['NL'], ship_thrust_rate['S'])),
            ctrl.Rule(aster_distance['L'] & theta_delta['NS'], (ship_turn['NS'], ship_thrust_rate['S'])),
            ctrl.Rule(aster_distance['L'] & theta_delta['Z'], (ship_turn['Z'], ship_thrust_rate['MS'])),
            ctrl.Rule(aster_distance['L'] & theta_delta['PS'], (ship_turn['PS'], ship_thrust_rate['S'])),
            ctrl.Rule(aster_distance['L'] & theta_delta['PL'], (ship_turn['PL'], ship_thrust_rate['S'])),
        ]
        self.moving_control = ctrl.ControlSystem(moving_rules)

    def setup_danger_fuzzy_system(self):
        collision_time = ctrl.Antecedent(np.arange(0, 10, 0.1), 'collision_time')
        asteroid_size = ctrl.Antecedent(np.arange(0, 5, 1), 'aster_size')
        collision_likely = ctrl.Consequent(np.arange(-1, 1, 0.1), 'collision_likely')

        collision_time['S'] = fuzz.trimf(collision_time.universe, [0, 0, 1])
        collision_time['MS'] = fuzz.trimf(collision_time.universe, [0, 2, 4])
        collision_time['M'] = fuzz.trimf(collision_time.universe, [2, 4, 6])
        collision_time['ML'] = fuzz.trimf(collision_time.universe, [4, 6, 8])
        collision_time['L'] = fuzz.trimf(collision_time.universe, [6, 10, 10])

        asteroid_size['S'] = fuzz.trimf(asteroid_size.universe, [0, 0, 1])
        asteroid_size['MS'] = fuzz.trimf(asteroid_size.universe, [0, 1, 2])
        asteroid_size['M'] = fuzz.trimf(asteroid_size.universe, [1, 2, 3])
        asteroid_size['ML'] = fuzz.trimf(asteroid_size.universe, [2, 3, 4])
        asteroid_size['L'] = fuzz.trimf(asteroid_size.universe, [3, 4, 4])

        collision_likely['N'] = fuzz.trimf(collision_likely.universe, [-1, -1, 0.0])
        collision_likely['Y'] = fuzz.trimf(collision_likely.universe, [0.0, 1, 1])

        danger_rules = [
            ctrl.Rule(collision_time['S'] & asteroid_size['S'], collision_likely['Y']),
            ctrl.Rule(collision_time['S'] & asteroid_size['MS'], collision_likely['Y']),
            ctrl.Rule(collision_time['S'] & asteroid_size['M'], collision_likely['Y']),
            ctrl.Rule(collision_time['S'] & asteroid_size['ML'], collision_likely['Y']),
            ctrl.Rule(collision_time['S'] & asteroid_size['L'], collision_likely['Y']),
            ctrl.Rule(collision_time['MS'] & asteroid_size['S'], collision_likely['N']),
            ctrl.Rule(collision_time['MS'] & asteroid_size['MS'], collision_likely['Y']),
            ctrl.Rule(collision_time['MS'] & asteroid_size['M'], collision_likely['Y']),
            ctrl.Rule(collision_time['MS'] & asteroid_size['ML'], collision_likely['Y']),
            ctrl.Rule(collision_time['MS'] & asteroid_size['L'], collision_likely['Y']),
            ctrl.Rule(collision_time['M'] & asteroid_size['S'], collision_likely['N']),
            ctrl.Rule(collision_time['M'] & asteroid_size['MS'], collision_likely['N']),
            ctrl.Rule(collision_time['M'] & asteroid_size['M'], collision_likely['Y']),
            ctrl.Rule(collision_time['M'] & asteroid_size['ML'], collision_likely['Y']),
            ctrl.Rule(collision_time['M'] & asteroid_size['L'], collision_likely['Y']),
            ctrl.Rule(collision_time['ML'] & asteroid_size['S'], collision_likely['N']),
            ctrl.Rule(collision_time['ML'] & asteroid_size['MS'], collision_likely['N']),
            ctrl.Rule(collision_time['ML'] & asteroid_size['M'], collision_likely['N']),
            ctrl.Rule(collision_time['ML'] & asteroid_size['ML'], collision_likely['Y']),
            ctrl.Rule(collision_time['ML'] & asteroid_size['L'], collision_likely['Y']),
            ctrl.Rule(collision_time['L'] & asteroid_size['S'], collision_likely['N']),
            ctrl.Rule(collision_time['L'] & asteroid_size['MS'], collision_likely['N']),
            ctrl.Rule(collision_time['L'] & asteroid_size['M'], collision_likely['N']),
            ctrl.Rule(collision_time['L'] & asteroid_size['ML'], collision_likely['N']),
            ctrl.Rule(collision_time['L'] & asteroid_size['L'], collision_likely['Y']),
        ]
        self.danger_control = ctrl.ControlSystem(danger_rules)

    def locate_n_closest_asteroids(self, n, ship_state, game_state):
        ship_pos_x = ship_state["position"][0]  # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]
        closest_asteroids = [None]

        for a in game_state["asteroids"]:
            curr_dist = math.sqrt((ship_pos_x - a["position"][0]) ** 2 + (ship_pos_y - a["position"][1]) ** 2)
            if closest_asteroids[0] is None:
                closest_asteroids[0] = dict(aster=a, dist=curr_dist)
            else:
                closest_asteroids.append(dict(aster=a, dist=curr_dist))

        closest_asteroids.sort(key=lambda x: x["dist"])
        closest_asteroids = closest_asteroids[:n]
        return closest_asteroids

    def target(self, ship_state, target_asteroid):
        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0]  # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]

        asteroid_ship_x = ship_pos_x - target_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - target_asteroid["aster"]["position"][1]

        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)
        asteroid_direction = math.atan2(target_asteroid["aster"]["velocity"][1], target_asteroid["aster"]["velocity"][0])  # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)
        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(
            target_asteroid["aster"]["velocity"][0] ** 2 + target_asteroid["aster"]["velocity"][1] ** 2)

        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * target_asteroid["dist"] * asteroid_vel * cos_my_theta2) ** 2 - (
                    4 * (asteroid_vel ** 2 - BULLET_SPEED ** 2) * target_asteroid["dist"])

        # Combine the Law of Cosines with the quadratic formula for solve for intercept time. Remember, there are two values produced.
        intrcpt1 = ((2 * target_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (
                    2 * (asteroid_vel ** 2 - BULLET_SPEED ** 2))
        intrcpt2 = ((2 * target_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (
                    2 * (asteroid_vel ** 2 - BULLET_SPEED ** 2))

        # Take the smaller intercept time, as long as it is positive; if not, take the larger one.
        if intrcpt1 > intrcpt2:
            if intrcpt2 >= 0:
                bullet_t = intrcpt2
            else:
                bullet_t = intrcpt1
        else:
            if intrcpt1 >= 0:
                bullet_t = intrcpt1
            else:
                bullet_t = intrcpt2

        # Calculate the intercept point. The work backwards to find the ship's firing angle my_theta1.
        intrcpt_x = target_asteroid["aster"]["position"][0] + target_asteroid["aster"]["velocity"][0] * bullet_t
        intrcpt_y = target_asteroid["aster"]["position"][1] + target_asteroid["aster"]["velocity"][1] * bullet_t

        my_theta1 = math.atan2((intrcpt_y - ship_pos_y), (intrcpt_x - ship_pos_x))

        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        shooting_theta = my_theta1 - ((math.pi / 180) * ship_state["heading"])

        # Wrap all angles to (-pi, pi)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        return shooting_theta, bullet_t

    def collision_imminent(self, ship_state, target_asteroid, ship_direction=0, thrust=0):
        ship_pos_x = ship_state["position"][0] + thrust
        ship_pos_y = ship_state["position"][1] + thrust

        asteroid_ship_x = ship_pos_x - target_asteroid["aster"]["position"][0]
        asteroid_ship_y = ship_pos_y - target_asteroid["aster"]["position"][1]

        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x) + ship_direction
        asteroid_direction = math.atan2(target_asteroid["aster"]["velocity"][1], target_asteroid["aster"]["velocity"][0])  # Velocity is a 2-element array [vx,vy].
        my_theta2 = asteroid_ship_theta - asteroid_direction
        cos_my_theta2 = math.cos(my_theta2)

        # Need the speeds of the asteroid and bullet. speed * time is distance to the intercept point
        asteroid_vel = math.sqrt(
            target_asteroid["aster"]["velocity"][0] ** 2 + target_asteroid["aster"]["velocity"][1] ** 2)

        # Determinant of the quadratic formula b^2-4ac
        targ_det = (-2 * target_asteroid["dist"] * asteroid_vel * cos_my_theta2) ** 2 - (4 * (asteroid_vel ** 2) * target_asteroid["dist"])

        if targ_det > 0:
            intrcpt1 = ((2 * target_asteroid["dist"] * asteroid_vel * cos_my_theta2) + math.sqrt(targ_det)) / (2 * (asteroid_vel ** 2))
            intrcpt2 = ((2 * target_asteroid["dist"] * asteroid_vel * cos_my_theta2) - math.sqrt(targ_det)) / (2 * (asteroid_vel ** 2))

            if intrcpt1 >= 0:
                return True, intrcpt1
            elif intrcpt2 >= 0:
                return True, intrcpt2

        return False, 0

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool]:
        closest_aster = self.locate_n_closest_asteroids(1, ship_state, game_state)[0]
        target_angle, bullet_time = self.target(ship_state, closest_aster)

        if closest_aster['dist'] <= 300:
            will_collide, collide_time = self.collision_imminent(ship_state, closest_aster)

            if will_collide and collide_time <= 10:
                danger_sim = ctrl.ControlSystemSimulation(self.danger_control, flush_after_run=1)
                danger_sim.input['collision_time'] = collide_time
                danger_sim.input['aster_size'] = closest_aster['aster']['size']
                danger_sim.compute()

                if danger_sim.output['collision_likely'] >= 0:
                    moving_sim = ctrl.ControlSystemSimulation(self.moving_control, flush_after_run=1)
                    moving_sim.input['aster_distance'] = closest_aster['dist']
                    moving_sim.input['theta_delta'] = target_angle
                    moving_sim.compute()

                    thrust = moving_sim.output['ship_thrust_rate']
                    turn_rate = moving_sim.output['ship_turn']
                    if turn_rate > 0:
                        turn_rate -= 180
                    elif turn_rate < 0:
                        turn_rate += 180
                    fire = 1
                    return thrust, turn_rate, fire

                    # for angle in range(0, 180, 30):
                    #     isSafe = 1
                    #     for a in self.locate_n_closest_asteroids(5, ship_state, game_state):
                    #         if turn_rate > 0: temp_turn_rate = turn_rate + angle
                    #         elif turn_rate < 0: temp_turn_rate = turn_rate - angle
                    #         if self.collision_imminent(ship_state, a, temp_turn_rate, thrust)[0]:
                    #             print('nah')
                    #             isSafe = 0
                    #             break
                    #     if isSafe:
                    #         fire = 1
                    #         if turn_rate > 0: turn_rate -= angle
                    #         elif turn_rate < 0: turn_rate += angle
                    #         return thrust, turn_rate, fire
                    # print('uhhhhhhh throw mine?')

        targeting_sim = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
        targeting_sim.input['bullet_time'] = bullet_time
        targeting_sim.input['theta_delta'] = target_angle
        targeting_sim.compute()

        thrust = 0
        turn_rate = targeting_sim.output['ship_turn']
        fire = targeting_sim.output['ship_fire']>=0

        self.eval_frames += 1
        return thrust, turn_rate, fire

    @property
    def name(self) -> str:
        return "Team Controller"