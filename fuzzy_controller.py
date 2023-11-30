from src.kesslergame import KesslerController
from typing import Dict, Tuple
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import numpy as np

BULLET_SPEED = 800

class FuzzyController(KesslerController):

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
            ctrl.Rule(bullet_time['L'], ship_fire['N']),
            ctrl.Rule(bullet_time['M'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['N'])),
            ctrl.Rule(bullet_time['M'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NL'], (ship_turn['NL'], ship_fire['N'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['NS'], (ship_turn['NS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['Z'], (ship_turn['Z'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PS'], (ship_turn['PS'], ship_fire['Y'])),
            ctrl.Rule(bullet_time['S'] & theta_delta['PL'], (ship_turn['PL'], ship_fire['N'])),
        ]
        self.targeting_control = ctrl.ControlSystem(target_rules)

    def setup_move_fuzzy_system(self):
        aster_distance = ctrl.Antecedent(np.arange(0, 300, 0.1), 'aster_distance')
        asteroid_size = ctrl.Antecedent(np.arange(0, 5, 1), 'aster_size')
        in_danger = ctrl.Antecedent(np.arange(-1, 1, 0.1), 'in_danger')
        ship_thrust_rate = ctrl.Consequent(np.arange(-200, 100, 0.1), 'ship_thrust_rate')

        aster_distance['S'] = fuzz.trimf(aster_distance.universe, [0, 0, 50])
        aster_distance['M'] = fuzz.trimf(aster_distance.universe, [40, 90, 140])
        aster_distance['L'] = fuzz.trimf(aster_distance.universe, [130, 300, 300])

        asteroid_size['S'] = fuzz.trimf(asteroid_size.universe, [0, 0, 2])
        asteroid_size['M'] = fuzz.trimf(asteroid_size.universe, [1, 2, 3])
        asteroid_size['L'] = fuzz.trimf(asteroid_size.universe, [2, 4, 4])

        in_danger['N'] = fuzz.trimf(in_danger.universe, [-1, -1, 0.0])
        in_danger['Y'] = fuzz.trimf(in_danger.universe, [0.0, 1, 1])

        ship_thrust_rate['emergency'] = fuzz.trimf(ship_thrust_rate.universe, [-200, -200, -120])
        ship_thrust_rate['NL'] = fuzz.trimf(ship_thrust_rate.universe, [-130, -100, -70])
        ship_thrust_rate['NS'] = fuzz.trimf(ship_thrust_rate.universe, [-75, -50, -40])
        ship_thrust_rate['Z'] = fuzz.trimf(ship_thrust_rate.universe, [-50, 0, 50])
        ship_thrust_rate['PS'] = fuzz.trimf(ship_thrust_rate.universe, [40, 50, 75])
        ship_thrust_rate['PL'] = fuzz.trimf(ship_thrust_rate.universe, [70, 100, 100])

        moving_rules = [
            ctrl.Rule(aster_distance['S'] & asteroid_size['S'] & in_danger['Y'], (ship_thrust_rate['NL'])),
            ctrl.Rule(aster_distance['S'], (ship_thrust_rate['emergency'])),
            ctrl.Rule(aster_distance['M'] & asteroid_size['S'] & in_danger['Y'], (ship_thrust_rate['NS'])),
            ctrl.Rule(aster_distance['M'] & asteroid_size['S'] & in_danger['N'], (ship_thrust_rate['PS'])),
            ctrl.Rule(aster_distance['M'] & asteroid_size['M'] & in_danger['Y'], (ship_thrust_rate['Z'])),
            ctrl.Rule(aster_distance['M'] & asteroid_size['M'] & in_danger['N'], (ship_thrust_rate['PS'])),
            ctrl.Rule(aster_distance['M'] & asteroid_size['L'], (ship_thrust_rate['NL'])),
            ctrl.Rule(aster_distance['L'], (ship_thrust_rate['PL'])),
        ]
        self.moving_control = ctrl.ControlSystem(moving_rules)

    def setup_danger_fuzzy_system(self):
        collision_time = ctrl.Antecedent(np.arange(0, 5, 0.1), 'collision_time')
        asteroid_size = ctrl.Antecedent(np.arange(0, 5, 1), 'aster_size')
        aster_distance = ctrl.Antecedent(np.arange(0, 300, 0.1), 'aster_distance')
        collision_likely = ctrl.Consequent(np.arange(-1, 1, 0.1), 'collision_likely')
        danger = ctrl.Consequent(np.arange(0, 10, 0.1), 'danger')

        collision_time['S'] = fuzz.trimf(collision_time.universe, [0, 0, 1])
        collision_time['M'] = fuzz.trimf(collision_time.universe, [0, 3, 5])
        collision_time['L'] = fuzz.trimf(collision_time.universe, [3, 5, 5])

        asteroid_size['S'] = fuzz.trimf(asteroid_size.universe, [0, 0, 1])
        asteroid_size['MS'] = fuzz.trimf(asteroid_size.universe, [0, 1, 2])
        asteroid_size['M'] = fuzz.trimf(asteroid_size.universe, [1, 2, 3])
        asteroid_size['ML'] = fuzz.trimf(asteroid_size.universe, [2, 3, 4])
        asteroid_size['L'] = fuzz.trimf(asteroid_size.universe, [3, 4, 4])

        aster_distance['S'] = fuzz.trimf(aster_distance.universe, [0, 0, 20])
        aster_distance['M'] = fuzz.trimf(aster_distance.universe, [10, 90, 140])
        aster_distance['L'] = fuzz.trimf(aster_distance.universe, [130, 300, 300])

        collision_likely['N'] = fuzz.trimf(collision_likely.universe, [-1, -1, 0.0])
        collision_likely['Y'] = fuzz.trimf(collision_likely.universe, [0.0, 1, 1])

        danger['emergency'] = fuzz.trimf(danger.universe, [9, 10, 10])
        danger['S'] = fuzz.trimf(danger.universe, [0, 0, 5])
        danger['M'] = fuzz.trimf(danger.universe, [2.5, 5, 7.5])
        danger['L'] = fuzz.trimf(danger.universe, [5, 10, 10])

        danger_rules = [
            ctrl.Rule(collision_time['S'] | aster_distance['S'], (collision_likely['Y'], danger['emergency'])),
            ctrl.Rule(collision_time['M'] & asteroid_size['S'], (collision_likely['N'], danger['M'])),
            ctrl.Rule(collision_time['M'] & asteroid_size['MS'], (collision_likely['N'], danger['M'])),
            ctrl.Rule(collision_time['M'] & aster_distance['M'], (collision_likely['Y'], danger['M'])),
            ctrl.Rule(asteroid_size['L'], (collision_likely['Y'], danger['L'])),
            ctrl.Rule(aster_distance['L'], (collision_likely['N'], danger['M'])),
        ]
        self.danger_control = ctrl.ControlSystem(danger_rules)

    def locate_n_closest_asteroids(self, n, ship_state, game_state):
        ship_pos_x = ship_state["position"][0]  # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1]
        closest_asteroids = [None]
        closest_asteroid_count = 0

        for a in game_state["asteroids"]:
            curr_dist = math.sqrt((ship_pos_x - a["position"][0]) ** 2 + (ship_pos_y - a["position"][1]) ** 2)
            if closest_asteroids[0] is None:
                closest_asteroids[0] = dict(aster=a, dist=curr_dist)
            else:
                closest_asteroids.append(dict(aster=a, dist=curr_dist))
            if curr_dist <= 10:
                closest_asteroid_count += 1

        closest_asteroids.sort(key=lambda x: x["dist"])
        closest_asteroids = closest_asteroids[:n]

        return closest_asteroids, closest_asteroid_count

    def locate_smallest_asteroid(self, closest_asteroids):
        closest_asteroids.sort(key=lambda x: x['aster']['size'])

        # Also gets the closest for multiple of the smallest size
        closest = closest_asteroids[0]['dist']
        smallest = closest_asteroids[0]
        print(smallest['aster']['size'])
        for a in closest_asteroids:
            if a['aster']['size'] == smallest['aster']['size']:
                if a['dist'] < closest:
                    smallest = a
                    closest = a['dist']
                    print(closest)
        return smallest

    def locate_biggest_asteroid(self, closest_asteroids):
        closest_asteroids.sort(key=lambda x: x['aster']['size'])
        # Also gets the closest for multiple of the biggest size
        closest = closest_asteroids[-1]['dist']
        biggest = closest_asteroids[-1]
        for a in closest_asteroids:
            if a['aster']['size'] == closest_asteroids[0]['aster']['size']:
                if a['dist'] < closest:
                    biggest = a
                    closest = a['dist']
        return biggest

    def target(self, ship_state, target_asteroid, thrust=0):
        # Find the closest asteroid (disregards asteroid velocity)
        ship_pos_x = ship_state["position"][0] + thrust  # See src/kesslergame/ship.py in the KesslerGame Github
        ship_pos_y = ship_state["position"][1] + thrust

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

    def escape(self,ship_state, map_size, n_closest_asteroids):
        angles = []
        n = len(n_closest_asteroids)

        for ast in n_closest_asteroids:
            rel_ax, rel_ay = self.rel_asteroid_pos(ship_state['position'], ast['position'], map_size)

            angle = (360 + math.atan2(rel_ay, rel_ax) * 180 / math.pi) % 360
            angles.append(angle)

        max_delta = 0
        ca = None
        angles = sorted(angles)
        for i in range(n):
            cur, next = angles[i], angles[(i + 1) % n]
            delta = (next - cur + 360) % 360
            if delta > max_delta:
                max_delta = delta
                ca = (cur + delta / 2) % 360

        if ca is None:
            ca = (angles[0] + 180) % 360

        # Lastly, find the difference betwwen firing angle and the ship's current orientation. BUT THE SHIP HEADING IS IN DEGREES.
        delta = ca - ship_state["heading"]
        if delta > 180:
            delta -= 360

        return delta

    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        def act(target):
            moving_sim = ctrl.ControlSystemSimulation(self.moving_control, flush_after_run=1)
            moving_sim.input['aster_distance'] = target['dist']
            moving_sim.input['aster_size'] = target['aster']['size']

            moving_sim.input['in_danger'] = collision_likely
            moving_sim.compute()

            thrust = moving_sim.output['ship_thrust_rate']

            target_angle, bullet_time = self.target(ship_state, target)
            targeting_sim = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
            targeting_sim.input['bullet_time'] = bullet_time
            targeting_sim.input['theta_delta'] = target_angle
            targeting_sim.compute()

            turn_rate = targeting_sim.output['ship_turn']
            fire = targeting_sim.output['ship_fire'] >= 0

            return thrust, turn_rate, fire

        closest_asters = self.locate_n_closest_asteroids(3, ship_state, game_state)[0]
        num_too_close = self.locate_n_closest_asteroids(3, ship_state, game_state)[1]
        # if num_too_close >= 5:

        most_dangerous_asteroid = closest_asters[0]
        highest_danger = 0

        for a in closest_asters:
            will_collide, collide_time = self.collision_imminent(ship_state, a)

            danger_sim = ctrl.ControlSystemSimulation(self.danger_control, flush_after_run=1)
            danger_sim.input['collision_time'] = collide_time
            danger_sim.input['aster_distance'] = a['dist']
            danger_sim.input['aster_size'] = a['aster']['size']
            danger_sim.compute()
            danger = danger_sim.output['danger']

            if danger > highest_danger:
                most_dangerous_asteroid = a
                highest_danger = danger

        will_collide, collide_time = self.collision_imminent(ship_state, most_dangerous_asteroid)
        danger_sim = ctrl.ControlSystemSimulation(self.danger_control, flush_after_run=1)
        danger_sim.input['collision_time'] = collide_time
        danger_sim.input['aster_distance'] = most_dangerous_asteroid['dist']
        danger_sim.input['aster_size'] = most_dangerous_asteroid['aster']['size']
        danger_sim.compute()
        collision_likely = danger_sim.output['collision_likely']

        print(collision_likely)
        if collision_likely > 0:
            thrust, turn_rate, fire = act(most_dangerous_asteroid)
        elif highest_danger >= 5:
            smallest_asteroid = self.locate_smallest_asteroid(closest_asters)
            thrust, turn_rate, fire = act(smallest_asteroid)
        else:
            largest_asteroid = self.locate_biggest_asteroid(closest_asters)
            thrust, turn_rate, fire = act(largest_asteroid)

        self.eval_frames += 1
        try:
            return thrust, turn_rate, fire
        except:
            return 0, turn_rate, fire

    @property
    def name(self) -> str:
        return "Team Controller"
