import torch
from ExponentialSchedule import ExponentialSchedule
from NinetyNineEnv import NinetyNineEnv
from DQN import DQN
from train import train_playing_dqn
import numpy as np

# Initialize environment and hyperparameters
env = NinetyNineEnv()
gamma = 0.95

num_steps = 500000
num_saves = 20

replay_size = 50_000
replay_prepopulate_steps = 50_000

batch_size = 64
exploration = ExponentialSchedule(1.0, 0.05, num_steps)

# Train the DQN
dqn_model, dqn_bid_model, returns, lengths, losses, bid_losses, testing_returns = train_playing_dqn(
    env,
    num_steps,
    num_saves=num_saves,
    replay_size=replay_size,
    replay_prepopulate_steps=replay_prepopulate_steps,
    batch_size=batch_size,
    exploration=exploration,
    gamma=gamma,
)

# Save the trained models
checkpoint_play = dqn_model.custom_dump()
checkpoint_bid = dqn_bid_model.custom_dump()

torch.save(checkpoint_bid, f'model_bid_test_2.pth')
torch.save(checkpoint_play, f'model_play_test_2.pth')

# Save testing returns
np.save('testing_return_take2.npy', testing_returns)

np.save('returns_take2.npy', returns)
np.save('lengths_take2.npy', lengths)
np.save('losses_take2.npy', losses)
np.save('bid_losses_take2.npy', bid_losses)
print("Training complete.")
