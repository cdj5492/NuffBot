import rlgym_sim

import obs
from logger import MLLogger
from terminal import FloorTouchedCondition
import reward as rwd
from startState import StartBalanceBall

from rlgym_sim.utils.reward_functions import CombinedReward
from rlgym_sim.utils.reward_functions.common_rewards import VelocityPlayerToBallReward, VelocityBallToGoalReward, LiuDistanceBallToGoalReward, \
    EventReward
from rlgym_sim.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_sim.utils.action_parsers import DiscreteAction



def build_rocketsim_env():

    spawn_opponents = False
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 1
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = DiscreteAction()
    terminal_conditions = [NoTouchTimeoutCondition(timeout_ticks), GoalScoredCondition(), FloorTouchedCondition()]

    rewards_to_combine = (rwd.LeaveFloorPenalty(),
                          rwd.DropBallPenalty(),
                          LiuDistanceBallToGoalReward(),
                          VelocityBallToGoalReward(),
                          EventReward(team_goal=10, concede=-10))
    reward_weights = (1.0, 1.0, 0.01, 1.0, 10.0)

    reward_fn = CombinedReward(reward_functions=rewards_to_combine,
                               reward_weights=reward_weights)

    #obs_builder = obs.NectoObsBuilder()
    obs_builder = obs.MLObs()

    state_setter = StartBalanceBall()

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

    n_proc = 16

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_rocketsim_env,
                      device="cpu",
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
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=200_000,
                      timestep_limit=1_000_000_000,
                      load_wandb=True,
                      wandb_run_name="balance_ppo_4",
                      checkpoint_load_folder="data/checkpoints/rlgym-ppo-run-1732048151671887100/59358150",
                      log_to_wandb=True)
    learner.learn()