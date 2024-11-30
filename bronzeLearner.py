import rlgym_sim

import obs
from logger import MLLogger
from terminal import FloorTouchedCondition
import reward as rwd
from startState import StartBalanceBall
from actionp import LookupAction

from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, LiuDistanceBallToGoalReward, \
    EventReward, FaceBallReward
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.state_setters import DefaultState



def build_rocketsim_env():

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition()]

    reward_fn = CombinedReward.from_zipped(
        (rwd.AirReward(), 0.15),
        (FaceBallReward(), 1.0),
        (rwd.SpeedTowardBallReward(), 5.0),
        (EventReward(touch=1), 50.0),
    )

    #obs_builder = obs.NectoObsBuilder()
    obs_builder = obs.MLObs()

    state_setter = DefaultState()

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
    import os
    metrics_logger = MLLogger()

    n_proc = 75

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
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.001,
                      ppo_epochs=1,
                      render=True,
                      render_delay=8.0/120.0,
                      add_unix_timestamp=False,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=200_000,
                      timestep_limit=1_000_000_000,
                      load_wandb=True,
                      wandb_run_name="bronzeLearner2",
                      checkpoint_load_folder=latest_checkpoint_dir,
                      log_to_wandb=True)
    learner.learn()