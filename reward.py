import numpy as np
from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils import RewardFunction
from rlgym_sim.utils.common_values import CAR_MAX_SPEED, CEILING_Z, BALL_RADIUS, SIDE_WALL_X, BACK_WALL_Y, BLUE_TEAM, ORANGE_TEAM, BALL_MAX_SPEED
import math

class BallZCoordinateReward(RewardFunction):
    def __init__(self, exponent=1):
        # Exponent should be odd so that negative y -> negative reward
        self.exponent = exponent

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return ((state.ball.position[2] - BALL_RADIUS) ** self.exponent)

class LeaveFloorPenalty(RewardFunction):
    def __init__(self):
        pass

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if not player.on_ground:
            return -1
        else:
            return 0

class DropBallPenalty(RewardFunction):
    def __init__(self):
        pass

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if not player.ball_touched:
            return -1
        else:
            return 0


class SpeedTowardBallReward(RewardFunction):
    # Default constructor
    def __init__(self):
        super().__init__()

    # Do nothing on game reset
    def reset(self, initial_state: GameState):
        pass

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        # Velocity of our player
        player_vel = player.car_data.linear_velocity
        
        # Difference in position between our player and the ball
        # When getting the change needed to reach B from A, we can use the formula: (B - A)
        pos_diff = (state.ball.position - player.car_data.position)
        
        # Determine the distance to the ball
        # The distance is just the length of pos_diff
        dist_to_ball = np.linalg.norm(pos_diff)
        
        # We will now normalize our pos_diff vector, so that it has a length/magnitude of 1
        # This will give us the direction to the ball, instead of the difference in position
        # Normalizing a vector can be done by dividing the vector by its length
        dir_to_ball = pos_diff / dist_to_ball

        # Use a dot product to determine how much of our velocity is in this direction
        # Note that this will go negative when we are going away from the ball
        speed_toward_ball = np.dot(player_vel, dir_to_ball)
        
        if speed_toward_ball > 0:
            # We are moving toward the ball at a speed of "speed_toward_ball"
            # The maximum speed we can move toward the ball is the maximum car speed
            # We want to return a reward from 0 to 1, so we need to divide our "speed_toward_ball" by the max player speed
            reward = speed_toward_ball / CAR_MAX_SPEED
            return reward
        else:
            # We are not moving toward the ball
            # Many good behaviors require moving away from the ball, so I highly recommend you don't punish moving away
            # We'll just not give any reward
            return 0
        
class AirReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass 

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if not player.on_ground:
            return 1
        else:
            return 0


MAX_TIME_IN_AIR = 1.75 # A rough estimate of the maximum reasonable aerial time

# reward for touching the ball in the air
class AirTouchReward(RewardFunction):
    def __init__(self):
        self.air_time = [0] * 8
        super().__init__()

    def reset(self, initial_state: GameState):
        self.air_time = [0] * 8
        pass 

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.on_ground:
            self.air_time[player.car_id] = 0
        else:
            self.air_time[player.car_id] += 8.0/120.0
        if player.ball_touched and not player.on_ground:
            # more rewrad the higher it is and the longer the player is in the air before touching it
            air_time_frac = min(self.air_time[player.car_id], MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR
            height_frac = state.ball.position[2] / CEILING_Z
            reward = min(air_time_frac, height_frac)
            return reward
        else:
            return 0

class ConserveBoostReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass 

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return math.sqrt(player.boost_amount)

class HitBallHardReward(RewardFunction):
    def __init__(self):
        self.previous_ball_velocity = np.zeros(3)
        super().__init__()

    def reset(self, initial_state: GameState):
        self.previous_ball_velocity = np.zeros(3)

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.ball_touched:
            velocity_change = np.linalg.norm(state.ball.linear_velocity - self.previous_ball_velocity) / BALL_MAX_SPEED
            self.previous_ball_velocity = state.ball.linear_velocity

            return velocity_change
        else:
            return 0

class StayOnTeamSideReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass 

    # Get the reward for a specific player, at the current state
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.team_num == BLUE_TEAM:
            if player.car_data.position[1] < state.ball.position[1]:
                return 1
            else:
                return -1
        else:
            if player.car_data.position[1] > state.ball.position[1]:
                return 1
            else:
                return -1

OCTANE_DRIBBLE_HEIGHT = 56.27 + BALL_RADIUS # 56.27 is the height of the top of the octane off the ground

class DribbleReward(RewardFunction):
    def __init__(self):
        # self.player_id_last_touched = 0
        super().__init__()

    def reset(self, initial_state: GameState):
        pass
        # self.player_id_last_touched = 0

    # get reward for balancing the ball at a certain height
    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.ball_touched:
            # only first two components
            relative_pos = state.ball.position[:2] - player.car_data.position[:2]
            mag = np.linalg.norm(relative_pos)
            if mag < BALL_RADIUS and state.ball.position[2] > player.car_data.position[2]:
                return 1
        return 0
        # if player.ball_touched:
        #     self.player_id_last_touched = player.car_id
            
        # if self.player_id_last_touched == player.car_id and abs(state.ball.position[2] - OCTANE_DRIBBLE_HEIGHT) < 20.0 and abs(state.ball.position[0]) < SIDE_WALL_X - 5*BALL_RADIUS and abs(state.ball.position[1]) < BACK_WALL_Y - 5*BALL_RADIUS:
        #     return 1
        # else:
        #     return 0

GOAL_WIDTH = 1786

class HitPostPenalty(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if abs(state.ball.position[0]) > GOAL_WIDTH/2:
            if (player.team_num == BLUE_TEAM and state.ball.position[1] > BACK_WALL_Y - 2*BALL_RADIUS) or (player.team_num == ORANGE_TEAM and state.ball.position[1] < -BACK_WALL_Y + 2*BALL_RADIUS):
                return -1
            else:
                return 0
        else:
            return 0

class NotMovingPenalty(RewardFunction):
    def __init__(self, epsilon_percent=0.05):
        self.epsilon_percent = epsilon_percent
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        mag = np.linalg.norm(player.car_data.linear_velocity)
        if mag < self.epsilon_percent * CAR_MAX_SPEED:
            return -1
        else:
            return 0

class FirstTouchReward(RewardFunction):
    def __init__(self):
        self.ball_touched = False
        super().__init__()

    def reset(self, initial_state: GameState):
        self.ball_touched = False

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.ball_touched and not self.ball_touched:
            self.ball_touched = True
            return 1
        else:
            return 0

class WinKickoffReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if state.ball.position[1] > 1.5 * BALL_RADIUS:
            if player.team_num == BLUE_TEAM:
                return 1
            else:
                return -1
        elif state.ball.position[1] < -1.5 * BALL_RADIUS:
            if player.team_num == ORANGE_TEAM:
                return 1
            else:
                return -1
        else:
            return 0

class MaximizeTimeBetweenFlipsReward(RewardFunction):
    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        if player.has_flip and not player.has_jump:
            # previous_action[
            return np.dot(player.car_data.forward(), np.array([0, 0, 1]))
        else:
            return 0

class CollectBoostPadReward(RewardFunction):
    def __init__(self):
        self.prev_boost_pickups = [0] * 8
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        reward = player.boost_pickups - self.prev_boost_pickups[player.car_id]
        self.prev_boost_pickups[player.car_id] = player.boost_pickups
        return reward

class VelocityReward(RewardFunction):
    def __init__(self):
        super().__init__()  

    def reset(self, initial_state: GameState):
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return np.linalg.norm(player.car_data.linear_velocity) / CAR_MAX_SPEED

class JumpOffWallReward(RewardFunction):
    def __init__(self):
        self.previous_has_jump = [False] * 8
        super().__init__()

    def reset(self, initial_state: GameState):
        self.previous_has_jump = [False] * 8

    def get_reward(self, player: PlayerData, state: GameState, previous_action) -> float:
        return_amt = 0
        if player.car_data.position[2] > CEILING_Z/4:
            if not player.has_jump and self.previous_has_jump[player.car_id]:
                return_amt = 1
        self.previous_has_jump[player.car_id] = player.has_jump
        return return_amt