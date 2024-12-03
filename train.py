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
    states, actions, rewards, next_states, dones = batch
    q_values = dqn_model(states)
    values = q_values.gather(1, actions).squeeze(-1)

    next_actions = dqn_model(next_states).argmax(dim=1, keepdim=True)
    next_q_values = dqn_target(next_states).gather(1, next_actions).squeeze(-1)

    target_values = rewards + gamma * next_q_values * (1 - dones.float())
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

    bid_state_size = 54
    play_state_size = 57

    dqn_bid_model = DQN(bid_state_size, 52, num_layers=3, hidden_dim=256)
    dqn_bid_target = DQN.custom_load(dqn_bid_model.custom_dump())

    dqn_model = DQN(play_state_size, 52, num_layers=3, hidden_dim=256)
    dqn_target = DQN.custom_load(dqn_model.custom_dump())

    optimizer_bid = torch.optim.RMSprop(dqn_bid_model.parameters(), lr=1e-4)
    optimizer_play = torch.optim.RMSprop(dqn_model.parameters(), lr=1e-4)

    torch.nn.utils.clip_grad_norm_(dqn_bid_model.parameters(), max_norm=10)
    torch.nn.utils.clip_grad_norm_(dqn_model.parameters(), max_norm=10)

    memory = ReplayMemory(replay_size, play_state_size)
    memory_bid = ReplayMemory(replay_size, bid_state_size)
    ReplayMemory.populate(env, replay_prepopulate_steps, memory_bid, memory)

    rewards = []
    testingReturns = []
    returns = []
    lengths = []
    losses = []

    i_episode = 0
    t_episode = 0
    G = 0

    old_dqn_model = DQN.custom_load(dqn_model.custom_dump())
    old_dqn_bid_model = DQN.custom_load(dqn_bid_model.custom_dump())

    pbar = tqdm.trange(num_steps)

    prev_phase = env.bidding_phase

    for t_total in pbar:
        current_dqn_model = dqn_model
        current_dqn_bid_model = dqn_bid_model

        if i_episode % 1000 == 0:
            testingReturns.append(test(env, current_dqn_model, current_dqn_bid_model))
            state = env.reset_game()

        if i_episode % 10000 == 0:
            old_dqn_model = DQN.custom_load(dqn_model.custom_dump())
            old_dqn_bid_model = DQN.custom_load(dqn_bid_model.custom_dump())

        if t_total % 500 == 0:
            dqn_bid_target.load_state_dict(dqn_bid_model.state_dict())
            dqn_target.load_state_dict(dqn_model.state_dict())

        eps = max(exploration.value(t_total), 0.01)

        if env.bidding_phase == 1:
            if env.current_player == 0:
                q_values = current_dqn_bid_model(torch.tensor(state, dtype=torch.float32))
            else:
                q_values = old_dqn_bid_model(torch.tensor(state, dtype=torch.float32))
        else:
            if env.current_player == 0:
                q_values = current_dqn_model(torch.tensor(state, dtype=torch.float32))
            else:
                q_values = old_dqn_model(torch.tensor(state, dtype=torch.float32))

        if random.random() > eps:
            possible_actions = env.possible_actions()
            num_actions = q_values.size(0)
            mask = torch.full((num_actions,), float('-inf'))
            mask[possible_actions] = 0

            masked_q_values = q_values + mask
            action = torch.argmax(masked_q_values).item()
        else:
            action = random.choice(env.possible_actions())

        prev_phase = env.bidding_phase
        next_state, reward, done, _ = env.step(action)

        if prev_phase == 1 and env.current_player == 0:
            memory_bid.add(state, action, reward, next_state, done)
        elif prev_phase == 0 and env.current_player == 0:
            memory.add(state, action, reward, next_state, done)

        state = next_state
        G += reward
        t_episode += 1

        if memory_bid.size >= batch_size and prev_phase == 1:
            for _ in range(5):
                batch = memory_bid.sample(batch_size)
                loss = train_dqn_batch(optimizer_bid, batch, dqn_bid_model, dqn_bid_target, gamma)
                losses.append(loss)

        if memory.size >= batch_size and prev_phase == 0:
            batch = memory.sample(batch_size)
            loss = train_dqn_batch(optimizer_play, batch, dqn_model, dqn_target, gamma)
            losses.append(loss)

        tau = 0.01
        for target_param, param in zip(dqn_target.parameters(), dqn_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(dqn_bid_target.parameters(), dqn_bid_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        if done:
            rewards.append(reward)
            lengths.append(t_episode)
            state = env.reset_game()
            pbar.set_description(
                f'Episode: {i_episode} | Steps: {t_episode + 1} | Return: {G:5.2f} | Epsilon: {eps:4.2f}'
            )
            G = 0
            t_episode = 0
            i_episode += 1

    return (
        dqn_target,
        dqn_bid_target,
        np.array(returns),
        np.array(lengths),
        np.array(losses),
        np.array(testingReturns)
    )

def test(env, current_dqn_model, current_dqn_bid_model, runs=5):
    total = 0
    for i in range(runs):
        state = env.reset_game()
        done = False
        total_reward = 0
        while not done:
            if env.bidding_phase == 1:
                q_values = current_dqn_bid_model(torch.tensor(state, dtype=torch.float32))
            else:
                q_values = current_dqn_model(torch.tensor(state, dtype=torch.float32))

            possible_actions = env.possible_actions()
            if len(possible_actions) == 0:
                break
            num_actions = q_values.size(0)
            mask = torch.full((num_actions,), float('-inf'))
            mask[possible_actions] = 0

            masked_q_values = q_values + mask
            if env.current_player != 0:
                action = random.choice(possible_actions)
            else:
                action = torch.argmax(masked_q_values).item()

            state, reward, done, _ = env.step(action)
            total_reward += reward

        total += total_reward

    return total / runs
