import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, MSELoss
from earl_pytorch import EARL
from earl_pytorch.model import EARL, NextGoalPredictor
from earl_pytorch.dataset.create_dataset import replay_to_dfs, convert_dfs, normalize
from torch.nn import Sequential
import numpy as np
import pickle
import pandas as pd

earl = EARL()

model = Sequential(earl, NextGoalPredictor(earl.n_dims))

# print model shape
print(model)

print("starting replay_to_dfs")
with open("processed-dataframes/000bd989-ec90-486d-ab99-264e72a1532e.pickle", 'rb') as f:
    dfs = pd.read_pickle(f)
print("starting convert_dfs")
x_data, y_data = convert_dfs(dfs, tensors=True)
normalize(x_data)
print(x_data[0].shape)
print(x_data[1].shape)
print(x_data[2].shape)


def rearrange_data(x_data, y_data):
    x_ball, x_boost, x_players = x_data
    (
        y_score, y_next_touch, y_collect, y_demo,
        y_throttle, y_steer, y_pitch, y_yaw,
        y_roll, y_jump, y_boost, y_handbrake
    ) = y_data

    num_frames = x_ball.shape[0]

    # Create new structure for x_data
    x_data_rearranged = [
        [x_ball[f], x_boost[f], x_players[f]] for f in range(num_frames)
    ]

    # Create new structure for y_data
    y_data_rearranged = [
        [
            y_score[f], y_next_touch[f], y_collect[f], y_demo[f],
            y_throttle[f], y_steer[f], y_pitch[f], y_yaw[f],
            y_roll[f], y_jump[f], y_boost[f], y_handbrake[f]
        ]
        for f in range(num_frames)
    ]

    # x_data_rearranged = np.asarray(x_data_rearranged)
    # y_data_rearranged = np.asarray(y_data_rearranged)
    # x_data_rearranged = torch.as_tensor(x_data_rearranged)
    # y_data_rearranged = torch.as_tensor(y_data_rearranged)


    return x_data_rearranged, y_data_rearranged
# x_data, y_data = rearrange_data(x_data, y_data)
# a = [torch.FloatTensor([1]).view(1, -1), torch.FloatTensor([2]).view(1, -1)]

# print(x_data[0][2][:2])

n_epochs = 10
learning_rate = 0.0002

# Define the optimizer and loss function
optimizer = Adam(model.parameters(), lr=0.001)
criterion = MSELoss()


# Training loop
for epoch in range(n_epochs):
    y_pred = model(*[torch.from_numpy(v).float().cuda() for v in x_data])
    loss = criterion(y_pred, y_data)   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")

