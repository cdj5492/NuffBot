import rlgym

from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils import common_values
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils import math
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition
import torch
import numpy as np


class CustomTerminalCondition(TerminalCondition):
    def reset(self, initial_state: GameState):
        pass

    def is_terminal(self, current_state: GameState) -> bool:
        return current_state.last_touch != -1


class CustomObsBuilder(ObsBuilder):
    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        obs = []
        
        #If this observation is being built for a player on the orange team, we need to invert all the physics data we use.
        inverted = player.team_num == common_values.ORANGE_TEAM
        
        if inverted:
            obs += state.inverted_ball.serialize()
        else:
            obs += state.ball.serialize()
            
        for player in state.players:
            if inverted:
                obs += player.inverted_car_data.serialize()
            else:
                obs += player.car_data.serialize()
        
        return np.asarray(obs, dtype=np.float32)

import numpy as np


class SpeedReward(RewardFunction):
    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # minimize distance to ball
        ball = state.ball
        car = player.car_data
        reward = -((car.position[0] - ball.position[0])**2 + (car.position[1] - ball.position[1])**2 + (car.position[2] - ball.position[2])**2)**0.5
        
        return reward
        
    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0

default_tick_skip = 8
physics_ticks_per_second = 120
ep_len_seconds = 20

max_steps = int(round(ep_len_seconds * physics_ticks_per_second / default_tick_skip))

#Make the default rlgym environment
env = rlgym.make(terminal_conditions=[CustomTerminalCondition(), TimeoutCondition(max_steps)], obs_builder=CustomObsBuilder(), reward_fn=SpeedReward())

while True:
    obs = env.reset()
    done = False

    while not done:
        #Here we sample a random action. If you have an agent, you would get an action from it here.
        action = env.action_space.sample() 

        next_obs, reward, done, gameinfo = env.step(action)

        print("Reward: " + str(reward))
        print("Done: " + str(done))
        print("Gameinfo: " + str(gameinfo))
        print("Next obs: " + str(next_obs))
        
        obs = next_obs

