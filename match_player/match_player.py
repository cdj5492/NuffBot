import time
import rlgym_sim
import pandas as pd
import os
import torch
import numpy as np
from discrete_policy import DiscreteFF
from actionp import LookupAction
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition
from rlgym_sim.utils.action_parsers import DiscreteAction, ActionParser
from gym.spaces import Discrete
from obs import MLObs

# You can get the OBS size from the rlgym-ppo console print-outs when you start your bot
OBS_SIZE = 92

# If you haven't set these, they are [256, 256, 256] by default
POLICY_LAYER_SIZES = [256, 256, 256]

class Agent:
    def __init__(self, brain_path):
        self.action_parser = LookupAction()
        self.num_actions = len(self.action_parser._lookup_table)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        
        device = torch.device("cpu")
        self.policy = DiscreteFF(OBS_SIZE, self.num_actions, POLICY_LAYER_SIZES, device)
        self.policy.load_state_dict(torch.load(brain_path, map_location=device))
        torch.set_num_threads(1)

    def act(self, state):
        with torch.no_grad():
            action_idx, probs = self.policy.get_action(state, True)
        
        action = np.array(self.action_parser.parse_actions([action_idx], None))
        if len(action.shape) == 2:
            if action.shape[0] == 1:
                action = action[0]
        
        if len(action.shape) != 1:
            raise Exception("Invalid action:", action)
        
        return action

if __name__ == "__main__":
    TPS = 120//8

    env = rlgym_sim.make(
            obs_builder=MLObs(),
            terminal_conditions=[GoalScoredCondition()],
            tick_skip=1,
            spawn_opponents=True
        )
    latest_checkpoint_dir = "C:/Users/Cole Johnson/OneDrive - rit.edu/machine_intelligence/final_project/data/checkpoints/rlgym-ppo-run/" + str(max(os.listdir("C:/Users/Cole Johnson/OneDrive - rit.edu/machine_intelligence/final_project/data/checkpoints/rlgym-ppo-run/"), key=lambda d: int(d)))
    nuff = Agent(latest_checkpoint_dir + "/PPO_POLICY.pt")
    greg = Agent("C:/Users/Cole Johnson/AppData/Local/RLBotGUIX/MyBots/connors_bot/src/PPO_POLICY.pt")

    while True:
        obs = env.reset()

        done = False
        steps = 0
        ep_reward = 0
        t0 = time.time()
        starttime = time.time()
        while not done:
            # state_wrapper = StateWrapper()
            actions_1 = nuff.act(obs[0])
            actions_2 = greg.act(obs[1])
            actions = [actions_1, actions_2]

            for i in range(8):
                obs, reward, done, state = env.step(actions)
                if done:
                    break

                env.render()
                ep_reward += reward[0]
                steps += 1

                # Sleep to keep the game in real time
                time.sleep(max(0, starttime + steps / (120) - time.time()))

        length = time.time() - t0
        print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, ep_reward))

