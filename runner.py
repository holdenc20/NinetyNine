import torch
from ExponentialSchedule import ExponentialSchedule
from NinetyNineEnv import NinetyNineEnv
from DQN import DQN
from train import train_playing_dqn
import numpy as np

env = NinetyNineEnv()
gamma = 0.99

num_steps = 15_000_00
num_saves = 5

replay_size = 200_000
replay_prepopulate_steps = 50_000

batch_size = 64
exploration = ExponentialSchedule(1.0, 0.05, 1_000_000)

dqn_models, returns, lengths, losses = train_playing_dqn(
    env,
    num_steps,
    num_saves=num_saves,
    replay_size=replay_size,
    replay_prepopulate_steps=replay_prepopulate_steps,
    batch_size=batch_size,
    exploration=exploration,
    gamma=gamma,
)

assert len(dqn_models) == num_saves
assert all(isinstance(value, DQN) for value in dqn_models.values())

checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
torch.save(checkpoint, f'checkpoint_{1}.pt')
#checkpoint = torch.load(f'checkpoint_{env.spec.id}.pt')

import matplotlib.pyplot as plt

def moving_average(data, *, window_size = 1000):
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

plt.plot(moving_average(returns))
plt.savefig('returns.png')
plt.plot(moving_average(lengths))
plt.savefig('lengths.png')
plt.plot(moving_average(losses))
plt.savefig('losses.png')

