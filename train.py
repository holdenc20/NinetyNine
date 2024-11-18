import torch
import torch.nn.functional as F
from tqdm import tqdm
import gymnasium as gym
import random
import numpy as np
import copy

import DQN
import ReplayMemory

def train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma) -> float:
    """Perform a single batch-update step on the given DQN model."""
    states, actions, rewards, next_states, dones = batch
    q_values = dqn_model(states)
    values = q_values.gather(1, actions).squeeze(-1)

    next_q_values = dqn_target(next_states).detach()
    max_next_q_values = next_q_values.max(dim=1)[0]

    target_values = rewards + gamma * max_next_q_values * (1 - dones.float())
    target_values = target_values.detach()

    loss = F.mse_loss(values, target_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train_dqn(
    env,
    num_steps,
    *,
    num_saves=5,
    replay_size,
    replay_prepopulate_steps=0,
    batch_size,
    exploration,
    gamma,
):
    state_size = env.observation_space.shape[0]

    dqn_model = DQN(state_size, env.action_space.n)
    dqn_target = DQN.custom_load(dqn_model.custom_dump())

    optimizer = torch.optim.Adam(dqn_model.parameters())

    memory = ReplayMemory(replay_size, state_size)
    memory.populate(env, replay_prepopulate_steps)

    rewards = []
    returns = []
    lengths = []
    losses = []

    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
    saved_models = {}

    i_episode = 0  # index of the current episode
    t_episode = 0  # time-step inside current episode

    state, info = env.reset()
    G=0

    pbar = tqdm.trange(num_steps)
    for t_total in pbar:

        # Save model
        if t_total in t_saves:
            model_name = f'{100 * t_total / num_steps:04.1f}'.replace('.', '_')
            saved_models[model_name] = copy.deepcopy(dqn_model)

        eps = exploration.value(t_total)
        if random.random() > eps:
            action = dqn_model(torch.tensor(state)).argmax().item()
        else:
            action = env.action_space.sample()


        next_state, reward, done, _, _ = env.step(action)

        memory.add(state, action, reward, next_state, done)

        state = next_state
        G += reward
        t_episode += 1

        if memory.size >= batch_size and t_total % 4 == 0:
            batch = memory.sample(batch_size)
            loss = train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma)
            losses.append(loss)

        if t_total % 10000 == 0:
            dqn_target.load_state_dict(dqn_model.state_dict())

        if done:
            returns.append(G)
            lengths.append(t_episode)
            state, info = env.reset()
            pbar.set_description(
                f'Episode: {i_episode} | Steps: {t_episode + 1} | Return: {G:5.2f} | Epsilon: {eps:4.2f}'
            )

            G = 0
            t_episode = 0
            i_episode += 1


    saved_models['100_0'] = copy.deepcopy(dqn_model)

    return (
        saved_models,
        np.array(returns),
        np.array(lengths),
        np.array(losses),
    )