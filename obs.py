import numpy as np
from rlgym_sim.utils.gamestates import GameState
import math
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils.obs_builders import ObsBuilder
from rlgym_sim.utils import common_values
from collections import Counter
from typing import Any
from rlgym_sim.utils.common_values import BLUE_TEAM, ORANGE_TEAM

class MLObsMirror(ObsBuilder):
    def __init__(self):
        lin_vel_coef=1/2300
        ang_vel_coef=1/math.pi
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z])
        ang_coef=1/math.pi
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball.copy()
            pads = state.inverted_boost_pads.copy()
        else:
            inverted = False
            ball = state.ball.copy()
            pads = state.boost_pads.copy()

        # mirror left side of the world to the right
        if player.car_data.position[0] < 0:
            player.car_data.position[0] = -player.car_data.position[0]
            player.car_data.linear_velocity[0] = -player.car_data.linear_velocity[0]
            player.car_data.angular_velocity[1] = -player.car_data.angular_velocity[1]
            player.car_data.angular_velocity[2] = -player.car_data.angular_velocity[2]


            ball.position[0] = -ball.position[0]
            ball.linear_velocity[0] = -ball.linear_velocity[0]
            ball.angular_velocity[1] = -ball.angular_velocity[1]
            ball.angular_velocity[2] = -ball.angular_velocity[2]

        obs = [ball.position * self.POS_COEF,
               (player.car_data.position - ball.position) * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action,
               pads]


        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, inverted)

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs, player: PlayerData, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            player_car.position * self.POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * self.LIN_VEL_COEF,
            player_car.angular_velocity * self.ANG_VEL_COEF,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car

class MLObs(ObsBuilder):
    def __init__(self):
        lin_vel_coef=1/2300
        ang_vel_coef=1/math.pi
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z])
        ang_coef=1/math.pi
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        # self.max_obs = None

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray):
        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position * self.POS_COEF,
               (player.car_data.position - ball.position) * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action,
               pads]

        self._add_player_to_obs(obs, player, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, inverted)

        obs.extend(allies)
        obs.extend(enemies)
        final = np.concatenate(obs)
        # if self.max_obs is None:
        #     self.max_obs = np.abs(final)
        # else:
        #     self.max_obs = np.maximum(self.max_obs, np.abs(final))
        # formatted so the decimal point is always alligned with the 3rd character
        # print(f"63rd entry {final[63]:>8.3f}")
        return final

    def _add_player_to_obs(self, obs, player: PlayerData, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            player_car.position * self.POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * self.LIN_VEL_COEF,
            player_car.angular_velocity * self.ANG_VEL_COEF,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car


BOOST_LOCATIONS = (
    (0.0, -4240.0, 70.0),
    (-1792.0, -4184.0, 70.0),
    (1792.0, -4184.0, 70.0),
    (-3072.0, -4096.0, 73.0),
    (3072.0, -4096.0, 73.0),
    (- 940.0, -3308.0, 70.0),
    (940.0, -3308.0, 70.0),
    (0.0, -2816.0, 70.0),
    (-3584.0, -2484.0, 70.0),
    (3584.0, -2484.0, 70.0),
    (-1788.0, -2300.0, 70.0),
    (1788.0, -2300.0, 70.0),
    (-2048.0, -1036.0, 70.0),
    (0.0, -1024.0, 70.0),
    (2048.0, -1036.0, 70.0),
    (-3584.0, 0.0, 73.0),
    (-1024.0, 0.0, 70.0),
    (1024.0, 0.0, 70.0),
    (3584.0, 0.0, 73.0),
    (-2048.0, 1036.0, 70.0),
    (0.0, 1024.0, 70.0),
    (2048.0, 1036.0, 70.0),
    (-1788.0, 2300.0, 70.0),
    (1788.0, 2300.0, 70.0),
    (-3584.0, 2484.0, 70.0),
    (3584.0, 2484.0, 70.0),
    (0.0, 2816.0, 70.0),
    (- 940.0, 3310.0, 70.0),
    (940.0, 3308.0, 70.0),
    (-3072.0, 4096.0, 73.0),
    (3072.0, 4096.0, 73.0),
    (-1792.0, 4184.0, 70.0),
    (1792.0, 4184.0, 70.0),
    (0.0, 4240.0, 70.0),
)


class NectoObsBuilder(ObsBuilder):
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    _norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)

    def __init__(self, tick_skip=8):
        super().__init__()
        self.demo_timers = None
        self.boost_timers = None
        self.current_state = None
        self.current_qkv = None
        self.current_mask = None
        self.tick_skip = tick_skip
        self._boost_locations = np.array(BOOST_LOCATIONS)
        self._boost_types = self._boost_locations[:, 2] > 72

    def reset(self, initial_state: GameState):
        self.demo_timers = Counter()
        self.boost_timers = np.zeros(len(initial_state.boost_pads))
        # self.current_state = initial_state

    def _maybe_update_obs(self, state: GameState):
        # Don't need to do this for RLBot
        # if state == self.current_state:  # No need to update
        #     return

        if self.boost_timers is None:
            self.reset(state)
        else:
            self.current_state = state

        qkv = np.zeros((1, 1 + len(state.players) + len(state.boost_pads), 24))  # Ball, players, boosts

        # Add ball
        n = 0
        ball = state.ball
        qkv[0, 0, 3] = 1  # is_ball
        qkv[0, 0, 5:8] = ball.position
        qkv[0, 0, 8:11] = ball.linear_velocity
        qkv[0, 0, 17:20] = ball.angular_velocity

        # Add players
        n += 1
        for player in state.players:
            if player.team_num == BLUE_TEAM:
                qkv[0, n, 1] = 1  # is_teammate
            else:
                qkv[0, n, 2] = 1  # is_opponent
            car_data = player.car_data
            qkv[0, n, 5:8] = car_data.position
            qkv[0, n, 8:11] = car_data.linear_velocity
            qkv[0, n, 11:14] = car_data.forward()
            qkv[0, n, 14:17] = car_data.up()
            qkv[0, n, 17:20] = car_data.angular_velocity
            qkv[0, n, 20] = player.boost_amount
            #             qkv[0, n, 21] = player.is_demoed
            qkv[0, n, 22] = player.on_ground
            qkv[0, n, 23] = player.has_flip

            # Different than training to account for varying player amounts
            if self.demo_timers[player.car_id] <= 0:
                self.demo_timers[player.car_id] = 3
            else:
                self.demo_timers[player.car_id] = max(self.demo_timers[player.car_id] - self.tick_skip / 120, 0)
            qkv[0, n, 21] = self.demo_timers[player.car_id] / 10
            n += 1

        # Add boost pads
        n = 1 + len(state.players)
        boost_pads = state.boost_pads
        qkv[0, n:, 4] = 1  # is_boost
        qkv[0, n:, 5:8] = self._boost_locations
        qkv[0, n:, 20] = 0.12 + 0.88 * self._boost_types  # Boost amount
        #         qkv[0, n:, 21] = boost_pads

        # Boost and demo timers
        new_boost_grabs = (boost_pads == 1) & (self.boost_timers == 0)  # New boost grabs since last frame
        self.boost_timers[new_boost_grabs] = 0.4 + 0.6 * (self._boost_locations[new_boost_grabs, 2] > 72)
        self.boost_timers *= boost_pads  # Make sure we have zeros right
        qkv[0, 1 + len(state.players):, 21] = self.boost_timers
        self.boost_timers -= self.tick_skip / 1200  # Pre-normalized, 120 fps for 10 seconds
        self.boost_timers[self.boost_timers < 0] = 0

        # Store results
        self.current_qkv = qkv / self._norm
        mask = np.zeros((1, qkv.shape[1]))
        mask[0, 1 + len(state.players):1 + len(state.players)] = 1
        self.current_mask = mask

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        self._maybe_update_obs(state)
        invert = player.team_num == ORANGE_TEAM

        qkv = self.current_qkv.copy()
        mask = self.current_mask.copy()

        main_n = state.players.index(player) + 1
        qkv[0, main_n, 0] = 1  # is_main
        if invert:
            qkv[0, :, (1, 2)] = qkv[0, :, (2, 1)]  # Swap blue/orange
            qkv *= self._invert  # Negate x and y values

        q = qkv[0, main_n, :]
        q = np.expand_dims(np.concatenate((q, previous_action), axis=0), axis=(0, 1))
        kv = qkv

        # Use relative coordinates
        kv[0, :, 5:11] -= q[0, 0, 5:11]
        return q, kv, mask