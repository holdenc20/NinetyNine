import torch
import torch.nn.functional as F
import tqdm
import gymnasium as gym
import random
import numpy as np
import copy

from DQN import DQN
from ReplayMemory import ReplayMemory

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


def train_playing_dqn(
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

    state = env.reset_game()
    state_size = len(state)
    #state_size = env.observation_space.shape[0]

    dqn_model = DQN(state_size, 52, num_layers=4, hidden_dim=256)
    dqn_target = DQN.custom_load(dqn_model.custom_dump())
    optimizer = torch.optim.Adam(dqn_model.parameters())

    memory = ReplayMemory(replay_size, state_size)
    memory.populate(env, replay_prepopulate_steps)

    rewards = []
    testingReturns = []
    returns = []
    lengths = []
    losses = []

    t_saves = np.linspace(0, num_steps, num_saves - 1, endpoint=False)
    saved_models = {}

    i_episode = 0  # index of the current episode
    t_episode = 0  # time-step inside current episode

    G=0

    old_dqn_model = DQN.custom_load(dqn_model.custom_dump())

    pbar = tqdm.trange(num_steps)
    for t_total in pbar:
        current_dqn_model = dqn_model
        if i_episode % 500 == 0:
            testingReturns.append(test(env, current_dqn_model, old_dqn_model))
            state = env.reset_game()

        if i_episode % 10000 == 0:
            old_dqn_model = DQN.custom_load(dqn_model.custom_dump())

        # Save model
        if t_total in t_saves:
            model_name = f'{100 * t_total / num_steps:04.1f}'.replace('.', '_')
            saved_models[model_name] = copy.deepcopy(dqn_model)

        eps = exploration.value(t_total)

        if env.current_player == 0:
            if random.random() > eps:
                q_values = current_dqn_model(torch.tensor(state, dtype=torch.float32))
                possible_actions = env.possible_actions()
                num_actions = q_values.size(0)
                mask = torch.full((num_actions,), float('-inf'))
                mask[possible_actions] = 0

                masked_q_values = q_values + mask

                action = torch.argmax(masked_q_values).item()
            else:
                action = random.choice(env.possible_actions())
        else:
            q_values = old_dqn_model(torch.tensor(state, dtype=torch.float32))
            possible_actions = env.possible_actions()
            num_actions = q_values.size(0)
            mask = torch.full((num_actions,), float('-inf'))
            mask[possible_actions] = 0

            masked_q_values = q_values + mask

            action = torch.argmax(masked_q_values).item()

        next_state, reward, done, _ = env.step(action)

        if env.current_player == 0:
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
            state = env.reset_game()
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
        np.array(testingReturns)
    )

def test(env, current_dqn_model, old_dqn_model):
    total = 0
    for i in range(100):
        state = env.reset_game()
        done = False
        total_reward = 0
        while not done:
            q_values = (
                current_dqn_model(torch.tensor(state, dtype=torch.float32))
                if env.current_player == 0
                else old_dqn_model(torch.tensor(state, dtype=torch.float32))
            )

            possible_actions = env.possible_actions()
            if len(possible_actions) == 0:
                break
            num_actions = q_values.size(0)
            mask = torch.full((num_actions,), float('-inf'))
            mask[possible_actions] = 0

            masked_q_values = q_values + mask
            action = torch.argmax(masked_q_values).item()

            state, reward, done, _ = env.step(action)
            total_reward += reward

        total += total_reward

    return total / 10
