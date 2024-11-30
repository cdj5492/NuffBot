import rlgym_sim

import os
import obs
from logger import MLLogger
from terminal import FloorTouchedCondition
import reward as rwd
import startState as ss
from actionp import LookupAction
from typing import List

from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, LiuDistanceBallToGoalReward, \
    EventReward, FaceBallReward
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition, TimeoutCondition
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils.state_setters import StateWrapper
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM, BACK_WALL_Y, BALL_RADIUS
import pandas as pd
import math
import random

in_folder = "processed-dataframes/"
files_in_folder = [f for f in os.listdir(in_folder) if os.path.isfile(os.path.join(in_folder, f))]

def build_rocketsim_env():
    global in_folder
    global files_in_folder

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 7
    match_timeout_seconds = 15
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))
    match_timeout_ticks = int(round(match_timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [TimeoutCondition(match_timeout_ticks), NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    reward_fn = CombinedReward.from_zipped(
        (EventReward(team_goal=1, concede=-1), 40),
        (VelocityBallToGoalReward(), 2.0),
        (rwd.SpeedTowardBallReward(), 0.1),
        (FaceBallReward(), 0.05),
        (rwd.AirReward(), 0.05),
        (rwd.AirTouchReward(), 5.0),
        (rwd.ConserveBoostReward(), 0.04),
        (rwd.HitBallHardReward(), 1.0),
        (rwd.StayOnTeamSideReward(), 0.2),
        (rwd.NotMovingPenalty(), 0.1),
        (rwd.DribbleReward(), 0.4),
        (rwd.HitPostPenalty(), 3.0),
    )

    #obs_builder = obs.NectoObsBuilder()
    obs_builder = obs.MLObs()

    
    state_setter = ss.CombinedStateSetter.from_zipped(
        (ss.StartReplay(in_folder, files_in_folder), 0.6),
        (DefaultState(), 0.15),
        (ss.AirRedirectSetup(), 0.05),
        (ss.AirDribbleSetup(), 0.2),
    )

    env = rlgym_sim.make(tick_skip=tick_skip,
                         team_size=team_size,
                         spawn_opponents=spawn_opponents,
                         terminal_conditions=terminal_conditions,
                         reward_fn=reward_fn,
                         obs_builder=obs_builder,
                         action_parser=action_parser,
                         state_setter=state_setter)

    return env

if __name__ == "__main__":
    from rlgym_ppo import Learner
    metrics_logger = MLLogger()

    n_proc = 85
    # n_proc = 1

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

        
    try:
        latest_checkpoint_dir = "data/checkpoints/rlgym-ppo-run/" + str(max(os.listdir("data/checkpoints/rlgym-ppo-run"), key=lambda d: int(d)))
    except:
        latest_checkpoint_dir = None

    learner = Learner(build_rocketsim_env,
                      device="auto",
                      n_proc=n_proc,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      ppo_batch_size=200000,
                      ts_per_iteration=200000,
                      exp_buffer_size=400000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.01,
                      ppo_clip_range=0.2,
                      ppo_epochs=2,
                      policy_lr=2e-4,
                      critic_lr=2e-4,
                      render=True,
                      render_delay=8.0/120.0,
                      add_unix_timestamp=False,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=2_000_000_000,
                      timestep_limit=1_000_000_000_000,
                      load_wandb=False,
                      wandb_run_name="aerialLearner1",
                      checkpoint_load_folder=latest_checkpoint_dir,
                      log_to_wandb=True)
    learner.learn()