import pandas as pd
from rlgym_sim.utils import common_values, math
import os
import random
import rlviser_py as vis
from rlgym_sim.utils.gamestates import PlayerData, GameState
import RocketSim as rs
import time

def load_replay_from_file(filename) -> pd.DataFrame:
    return pd.read_pickle(filename)

def get_player_names(game_data: pd.DataFrame):
    names = (game_data.columns[0][0], game_data.columns[22][0])
    # check posiiton of the cars during the first frame to see which team they are on
    if game_data.loc[1, (names[0], 'pos_y')] < 0:
        return names
    else:
        return names[::-1]

def set_game_state_at_step(arena, step, game_data, player_names):
    # get ball position
    try:
        cars = arena.get_cars()
        ball = rs.BallState(
            pos = rs.Vec(
                x=game_data.loc[step, ('ball', 'pos_x')],
                y=game_data.loc[step, ('ball', 'pos_y')],
                z=game_data.loc[step, ('ball', 'pos_z')]
            ),
            vel = rs.Vec(
                x=game_data.loc[step, ('ball', 'vel_x')],
                y=game_data.loc[step, ('ball', 'vel_y')],
                z=game_data.loc[step, ('ball', 'vel_z')]
            )
        )
        ball.ang_vel = rs.Vec(
            x=game_data.loc[step, ('ball', 'ang_vel_x')] / 1000.0,
            y=game_data.loc[step, ('ball', 'ang_vel_y')] / 1000.0,
            z=game_data.loc[step, ('ball', 'ang_vel_z')] / 1000.0
        )
        arena.ball.set_state(ball)

        for car, name in zip(cars, player_names):
            car_state = rs.CarState()
            car_state.pos=rs.Vec(
                x=game_data.loc[step, (name, 'pos_x')],
                y=game_data.loc[step, (name, 'pos_y')],
                z=game_data.loc[step, (name, 'pos_z')]
            )
            car_state.vel=rs.Vec(
                x=game_data.loc[step, (name, 'vel_x')],
                y=game_data.loc[step, (name, 'vel_y')],
                z=game_data.loc[step, (name, 'vel_z')]
            )
            car_state.ang_vel=rs.Vec(
                x=game_data.loc[step, (name, 'ang_vel_x')] / 1000.0,
                y=game_data.loc[step, (name, 'ang_vel_y')] / 1000.0,
                z=game_data.loc[step, (name, 'ang_vel_z')] / 1000.0 
            )

            mtx = math.euler_to_rotation([
                -game_data.loc[step, (name, 'rot_x')],
                game_data.loc[step, (name, 'rot_y')],
                -game_data.loc[step, (name, 'rot_z')]
            ])
            rot = rs.RotMat(*mtx.transpose().flatten())

            car_state.rot_mat = rot
            car_state.boost = game_data.loc[step, (name, 'boost')] * 100.0 / 255.0
            if car_state.boost == float('nan'):
                car_state.boost = 0.0 
            # car_state.is_jumping = game_data.loc[step, (name, 'jump_active')]

            car.set_state(car_state)
    except Exception as e:
        print(e)



def play_replay(game_data):
    
    game_data = load_replay_from_file(random_replay)

    player_names = get_player_names(game_data)

    steps = 1
    start_time = time.time()

    game_mode = rs.GameMode.SOCCAR
    arena = rs.Arena(game_mode)
    tick_rate = 30
    tick_ratio = tick_rate/120
    arena.add_car(rs.Team.BLUE) 
    arena.add_car(rs.Team.ORANGE)

    while steps < game_data.shape[0]:
        if vis.get_game_paused():
            arena.step(1)
        else:
            set_game_state_at_step(arena, steps, game_data, player_names)

        car_data = [
            (car.id, car.team, car.get_config(), car.get_state())
            for car in arena.get_cars()
        ]

        ball = arena.ball.get_state()

        vis.render(steps, tick_rate, game_mode, [True] * 34, ball, car_data)

        # while vis.get_game_paused():
        #     time.sleep(0.1)

        # sleep to simulate running real time (it will run a LOT after otherwise)
        time.sleep(max(0, start_time + steps / tick_rate - time.time()))
        steps += 1

    vis.quit()

if __name__ == "__main__":
    in_folder = "processed-dataframes/"
    files_in_folder = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, f))]

    random_replay = in_folder + random.choice(files_in_folder)

    play_replay(random_replay)
