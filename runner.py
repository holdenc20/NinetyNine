import torch
from ExponentialSchedule import ExponentialSchedule
from NinetyNineEnv import NinetyNineEnv
from DQN import DQN
from train import train_playing_dqn
import numpy as np

env = NinetyNineEnv()
gamma = 0.99

num_steps = 5000
num_saves = 20

replay_size = 200_000
replay_prepopulate_steps = 50_000

batch_size = 64
exploration = ExponentialSchedule(1.0, 0.05, 5000)

dqn_models, returns, lengths, losses, testing_returns = train_playing_dqn(
    env,
    num_steps,
    num_saves=num_saves,
    replay_size=replay_size,
    replay_prepopulate_steps=replay_prepopulate_steps,
    batch_size=batch_size,
    exploration=exploration,
    gamma=gamma,
)

#assert len(dqn_models) == num_saves
#assert all(isinstance(value, DQN) for value in dqn_models.values())

checkpoint = {key: dqn.custom_dump() for key, dqn in dqn_models.items()}
torch.save(checkpoint, f'model_testLargerNET.pth')
#checkpoint = torch.load(f'checkpoint_{env.spec.id}.pt')

import matplotlib.pyplot as plt

"""
linspace_returns = np.linspace(0, len(returns)-1, len(returns), endpoint=True)
plt.plot(linspace_returns, returns)
plt.savefig('returns.png')

linspace_lengths = np.linspace(0, len(lengths)-1, len(lengths), endpoint=True)
plt.plot(linspace_lengths, lengths)
plt.savefig('lengths.png')

linspace_losses = np.linspace(0, len(losses)-1, len(losses), endpoint=True)
plt.plot(linspace_losses, losses)
plt.savefig('losses.png')
"""

np.save('testing_returns2.npy', testing_returns)

linspace_testing_returns = np.linspace(0, len(testing_returns)-1, len(testing_returns), endpoint=True)

window_size = 10
averaged_testing_returns = np.mean(testing_returns[:len(testing_returns) - len(testing_returns) % window_size].reshape(-1, window_size), axis=1)

linspace_averaged = np.linspace(0, len(averaged_testing_returns)-1, len(averaged_testing_returns), endpoint=True)

plt.plot(linspace_averaged, averaged_testing_returns)
plt.savefig('testing_returns_averaged2.png')
