from earl_pytorch.util.analyze import plot_replay
from earl_pytorch import EARL
from earl_pytorch.model import EARL, NextGoalPredictor
from torch.nn import Sequential

earl = EARL()

model = Sequential(earl, NextGoalPredictor(earl.n_dims))

plot_replay("ranked-duels-raw/0a0b9ac9-e014-4693-8240-847d06107728.replay", model)