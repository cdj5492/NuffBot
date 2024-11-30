import numpy as np
from rlgym_ppo.util import MetricsLogger
from rlgym_sim.utils.gamestates import GameState

class MLLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [game_state.players[0].car_data.linear_velocity,
                game_state.players[0].car_data.rotation_mtx(),
                game_state.players[0].boost_amount]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        avg_boost_amount = 0
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
            avg_boost_amount += metric_array[2]
        avg_linvel /= len(collected_metrics)
        avg_boost_amount /= len(collected_metrics)
        report = {"x_vel":avg_linvel[0],
                  "y_vel":avg_linvel[1],
                  "z_vel":avg_linvel[2],
                  "average boost":avg_boost_amount,
                  "Cumulative Timesteps":cumulative_timesteps}
        wandb_run.log(report)
