from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils import StateSetter
from rlgym_sim.utils.common_values import BALL_RADIUS, CEILING_Z, GRAVITY_Z
from rlgym_sim.utils.common_values import SIDE_WALL_X, BACK_WALL_Y, BALL_MAX_SPEED
from rlgym_sim.utils.state_setters import DefaultState
from typing import List
import numpy as np
import pandas as pd
import random
import math

class StartBalanceBall(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        # random position in the world between the back wall and the side wall
        x = random.uniform(-SIDE_WALL_X, SIDE_WALL_X)
        y = random.uniform(-BACK_WALL_Y, BACK_WALL_Y)
        
        
        car = state_wrapper.cars[0]
        pos = [x, y, 17]
        yaw = math.pi * random.randint(0, 360) / 180
        # set car state values
        car.set_pos(*pos)
        car.set_rot(yaw=yaw)
        car.boost = 0.33

        pos = [x, y, 150]
        state_wrapper.ball.set_pos(*pos)

def load_replay_from_file(filename) -> pd.DataFrame:
    # print(f"Loading: {filename}")
    return pd.read_pickle(filename)

def get_player_names(game_data: pd.DataFrame):
    names = (game_data.columns[0][0], game_data.columns[22][0])
    # check posiiton of the cars during the first frame to see which team they are on
    if game_data.loc[1, (names[0], 'pos_y')] < 0:
        return names
    else:
        return names[::-1]

def verify_no_nan(arr: List[float]):
    for val in arr:
        if math.isnan(val):
            raise ValueError
class StartReplay(StateSetter):
    def __init__(self, in_folder, files_in_folder):
        self.in_folder = in_folder
        self.files_in_folder = files_in_folder
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        got_good_start = False
        while not got_good_start:
            replay_file_name = self.in_folder + random.choice(self.files_in_folder)
            replay = load_replay_from_file(replay_file_name)
            names = get_player_names(replay)

            try:
                random_step = random.randint(1, len(replay) - 2)
                ball = state_wrapper.ball
                pos = [
                    replay.loc[random_step, ('ball', 'pos_x')],
                    replay.loc[random_step, ('ball', 'pos_y')],
                    replay.loc[random_step, ('ball', 'pos_z')]
                ]
                verify_no_nan(pos)
                ball.set_pos(*pos)

                # inside the goal
                if abs(ball.position[1]) > BACK_WALL_Y + BALL_RADIUS or (abs(ball.position[0]) <= 1 and abs(ball.position[1]) <= 1):
                    continue

                vel = [
                    replay.loc[random_step, ('ball', 'vel_x')] * 36000.0 / 229369.0,
                    replay.loc[random_step, ('ball', 'vel_y')] * 36000.0 / 229369.0,
                    replay.loc[random_step, ('ball', 'vel_z')] * 36000.0 / 229369.0
                ]
                # mag = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
                # if mag > BALL_MAX_SPEED:
                #     print("ball speed too fast:", mag, "max:", BALL_MAX_SPEED)
                verify_no_nan(vel)
                # check magnitude of velocity to make sure it's not too fast
                # mag = math.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
                # if mag > BALL_MAX_SPEED:
                #     print("too fast:", mag, "max:", BALL_MAX_SPEED)
                ball.set_lin_vel(*vel)

                ang_vel = [
                    replay.loc[random_step, ('ball', 'ang_vel_x')] / 1000.0,
                    replay.loc[random_step, ('ball', 'ang_vel_y')] / 1000.0,
                    replay.loc[random_step, ('ball', 'ang_vel_z')] / 1000.0
                ]
                verify_no_nan(ang_vel)
                ball.set_ang_vel(*ang_vel)

                for car, name in zip(state_wrapper.cars, names):
                    pos = [
                        replay.loc[random_step, (name, 'pos_x')],
                        replay.loc[random_step, (name, 'pos_y')],
                        replay.loc[random_step, (name, 'pos_z')]
                    ]
                    verify_no_nan(pos)
                    car.set_pos(*pos)

                    rot = [
                        -replay.loc[random_step, (name, 'rot_x')],
                        replay.loc[random_step, (name, 'rot_y')],
                        -replay.loc[random_step, (name, 'rot_z')]
                    ]
                    verify_no_nan(rot)
                    car.set_rot(*rot)

                    vel = [
                        replay.loc[random_step, (name, 'vel_x')],
                        replay.loc[random_step, (name, 'vel_y')],
                        replay.loc[random_step, (name, 'vel_z')]
                    ]
                    verify_no_nan(vel)
                    car.set_lin_vel(*vel)

                    ang_vel = [
                        replay.loc[random_step, (name, 'ang_vel_x')] / 1000.0,
                        replay.loc[random_step, (name, 'ang_vel_y')] / 1000.0,
                        replay.loc[random_step, (name, 'ang_vel_z')] / 1000.0
                    ]
                    verify_no_nan(ang_vel)
                    car.set_ang_vel(*ang_vel)
                    car.boost = replay.loc[random_step, (name, 'boost')] / 255.0
                    if math.isnan(car.boost):
                        raise ValueError

                got_good_start = True
            except:
                pass

class CombinedStateSetter(StateSetter):
    def __init__(self, setters=[], probs=[]):
        self.setters = setters
        self.probs = probs
        super().__init__()
    
    @classmethod
    def from_zipped(cls, *args):
        setters, probs = list(zip(*args))

        # normalize probabilities
        total_prob = sum(probs)
        probs = [prob / total_prob for prob in probs]
        return cls(setters, probs)

    def reset(self, state_wrapper: StateWrapper):
        selected_setter = np.random.choice(self.setters, p=self.probs)
        selected_setter.reset(state_wrapper)

class AirRedirectSetup(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            # give all cars 100 boost
            car.boost = 1

            # put both cars somewhere in the air facing upwards
            random_pos = [
                random.uniform(-SIDE_WALL_X + BALL_RADIUS, SIDE_WALL_X - BALL_RADIUS),
                random.uniform(-BACK_WALL_Y + 1152, BACK_WALL_Y - 1152),
                random.uniform(CEILING_Z/2, CEILING_Z - 4 * BALL_RADIUS)
            ]
            car.set_pos(*random_pos)

            # facing upwards
            car.set_rot(math.pi/2, 0, random.uniform(0, 2 * math.pi))

        # spawn the ball somewhere random
        random_pos = [
            random.uniform(-SIDE_WALL_X + BALL_RADIUS, SIDE_WALL_X - BALL_RADIUS),
            random.uniform(-BACK_WALL_Y + 1152, BACK_WALL_Y - 1152),
            random.uniform(CEILING_Z + BALL_RADIUS, CEILING_Z - 4 * BALL_RADIUS)
        ]
        state_wrapper.ball.set_pos(*random_pos)

        # pick a random car
        random_car = random.choice(state_wrapper.cars)

        
        tof = random.uniform(3.5, 7) / 120

        # add random noise to where the target is
        car_pos = random_car.position + np.random.uniform(-4*BALL_RADIUS, 4*BALL_RADIUS, 3)
        ball_pos = state_wrapper.ball.position
        displacement = car_pos - ball_pos 
        # calculate the velocity to give the to hit that car
        velocity = [
            displacement[0] / tof,
            displacement[1] / tof,
            (displacement[2] - .5 * GRAVITY_Z * tof * tof) / tof,
        ]

        state_wrapper.ball.set_lin_vel(*velocity)

class AirDribbleSetup(StateSetter):
    def __init__(self):
        super().__init__()
        
    def reset(self, state_wrapper: StateWrapper):
        for car in state_wrapper.cars:
            # give all cars 100 boost
            car.boost = 1

            # put both cars somewhere in the air facing upwards
            random_pos = [
                random.uniform(-SIDE_WALL_X + BALL_RADIUS, SIDE_WALL_X - BALL_RADIUS),
                random.uniform(-BACK_WALL_Y + 1152, BACK_WALL_Y - 1152),
                random.uniform(CEILING_Z/2, CEILING_Z - 4 * BALL_RADIUS)
            ]
            car.set_pos(*random_pos)

            # facing upwards
            car.set_rot(math.pi/2, 0, random.uniform(0, 2 * math.pi))

        # pick a random car
        random_car = random.choice(state_wrapper.cars)

        # put the ball on top of the car
        state_wrapper.ball.set_pos(*(random_car.position + np.array([0, 0, 118/2 + BALL_RADIUS])))