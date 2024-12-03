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
    timeout_seconds = 10
    match_timeout_seconds = 25
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))
    match_timeout_ticks = int(round(match_timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [TimeoutCondition(match_timeout_ticks), NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    reward_fn = CombinedReward.from_zipped(
        (EventReward(team_goal=1, concede=-1), 2.0 * 100),
        # (VelocityBallToGoalReward(), 0.64 * 100 / match_timeout_ticks),
        # (rwd.SpeedTowardBallReward(), 0.64 * 100 / match_timeout_ticks),
        (rwd.AirReward(), 0.05 * 100 / match_timeout_ticks),
        (FaceBallReward(), 0.32 * 100 / match_timeout_ticks),
        (rwd.AirTouchReward(), 0.1 * 100),
        (rwd.ConserveBoostReward(), 0.5 * 100 / match_timeout_ticks),
        (rwd.HitBallHardReward(), 0.1 * 100),
        (rwd.StayOnTeamSideReward(), 0.2 * 100 / match_timeout_ticks),
        (rwd.NotMovingPenalty(), 0.05 * 100 / match_timeout_ticks),
        (rwd.DribbleReward(), 0.5 * 100 / match_timeout_ticks),
        (rwd.HitPostPenalty(), 0.1 * 100 / match_timeout_ticks),
        (rwd.FirstTouchReward(), 0.3 * 100),
        (rwd.MaximizeTimeBetweenFlipsReward(), 0.2 * 100 / match_timeout_ticks),
        (rwd.CollectBoostPadReward(), .1 * 100 / 5),
        (rwd.VelocityReward(), 0.35 * 100 / match_timeout_ticks),
        (rwd.JumpOffWallReward(), 0.04 * 100)
    )

    #obs_builder = obs.NectoObsBuilder()
    obs_builder = obs.MLObs()

    
    state_setter = ss.CombinedStateSetter.from_zipped(
        (ss.StartReplay(in_folder, files_in_folder), 0.2),
        (DefaultState(), 0.65),
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
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=100000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.001,
                      ppo_clip_range=0.2,
                      ppo_epochs=1,
                      policy_lr=1e-4,
                      critic_lr=1e-4,
                      render=True,
                      render_delay=8.0/120.0,
                      add_unix_timestamp=False,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=20_000_000,
                      timestep_limit=1_000_000_000_000,
                      load_wandb=True,
                      wandb_run_name="NormalizedLearner1",
                      checkpoint_load_folder=latest_checkpoint_dir,
                      log_to_wandb=True)
    learner.learn()