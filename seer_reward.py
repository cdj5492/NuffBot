import numpy as np
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils import RewardFunction
import rlgym_sim.utils.common_values as cv
import math

class GoalScoredReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        old_goals_scoredZ