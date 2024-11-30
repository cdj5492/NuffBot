import time
import random
import rlgym_sim
import pandas as pd
import os
from rlgym_sim.utils.state_setters import StateWrapper

def load_replay_from_file(filename) -> pd.DataFrame:
    return pd.read_pickle(filename)

def get_player_names(game_data: pd.DataFrame):
    return (game_data.columns[0][0], game_data.columns[22][0])




def play_replay(in_file):
    game_data = load_replay_from_file(in_file)

    player_names = get_player_names(game_data)

    TPS = 120//8

    env = rlgym_sim.make(
            terminal_conditions=[],
            tick_skip=TPS,
            spawn_opponents=True
        )

    while True:
        obs = env.reset()

        done = False
        steps = 0
        ep_reward = 0
        t0 = time.time()
        starttime = time.time()
        while not done:
            # state_wrapper = StateWrapper()
            actions_1 = env.action_space.sample()
            actions_2 = env.action_space.sample()
            actions = [actions_1, actions_2]
            new_obs, reward, done, state = env.step(actions)
            try:
                for car, name in zip(state.cars, player_names):
                    pos = [game_data.loc[steps, (name, 'pos_x')], game_data.loc[steps, (name, 'pos_y')], game_data.loc[steps, (name, 'pos_z')]]
                    rot = [game_data.loc[steps, (name, 'rot_x')], game_data.loc[steps, (name, 'rot_y')], game_data.loc[steps, (name, 'rot_z')]]
                    car.set_pos(*pos)
                    car.set_rot(*rot)
            except Exception as e:
                print("error: " + str(e) + " skipping frame " + str(steps))

            env.render()
            ep_reward += reward[0]
            steps += 1

            # Sleep to keep the game in real time
            time.sleep(max(0, starttime + steps / TPS - time.time()))

        length = time.time() - t0
        print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, ep_reward))

if __name__ == "__main__":
    in_folder = "processed-dataframes/"
    files_in_folder = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, f))]

    random_replay = in_folder + random.choice(files_in_folder)

    play_replay(random_replay)