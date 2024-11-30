from rlgym_sim.utils.gamestates import PlayerData, GameState
from rlgym_sim.utils import TerminalCondition
from rlgym_sim.utils.common_values import BALL_RADIUS

class FloorTouchedCondition(TerminalCondition):

    def __init__(self):
        super().__init__()
        self.last_touch = None

    def reset(self, initial_state: GameState):
        self.last_touch = initial_state.last_touch

    def is_terminal(self, state: GameState) -> bool:
        return ((state.ball.position[2] - BALL_RADIUS - 0.1) <= 0)

class KickoffWonCondition(TerminalCondition):

    def __init__(self):
        super().__init__()
        self.last_touch = None

    def reset(self, initial_state: GameState):
        self.last_touch = initial_state.last_touch

    def is_terminal(self, state: GameState) -> bool:
        return abs(state.ball.position[1]) > 2 * BALL_RADIUS
            