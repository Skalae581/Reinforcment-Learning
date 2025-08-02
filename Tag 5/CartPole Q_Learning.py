# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 10:42:52 2025

@author: TAKO
"""

import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# --- Hyperparameter ---
EPISODES = 500
LR = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 10000
EPS_START = 1.0
EPS_MIN = 0.05
EPS_DECAY = 0.99
TARGET_UPDATE = 10
WARMUP_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Neural Network für Q-Funktion ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))
    def __len__(self):
        return len(self.buffer)

# --- Training Schritt (Double DQN) ---
def train_step(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Q(s,a)
    q_values = policy_net(states).gather(1, actions)

    # Double DQN: action via policy_net, value via target_net
    next_actions = policy_net(next_states).argmax(1, keepdim=True)
    next_q_values = target_net(next_states).gather(1, next_actions).squeeze()

    # Target
    target = rewards + GAMMA * next_q_values * (1 - dones)

    # Loss
    loss = nn.MSELoss()(q_values.squeeze(), target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Haupttraining ---
def dqn_train(env_name="CartPole-v1"):
   # env = gym.make(env_name)  # Kein render_mode, sonst zu langsam
    render = False  # Training schneller ohne Rendern

    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPS_START
    rewards = []
    step_count = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            # --- Epsilon-greedy ---
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax().item()

            # Schritt
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Optional: Reward clippen (hilft Stabilität)
            reward = np.clip(reward, -1, 1)

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1

            # Training erst nach Warmup starten
            if step_count > WARMUP_STEPS:
                train_step(policy_net, target_net, memory, optimizer)

        # Target-Net alle n Episoden synchronisieren
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Epsilon decay
        if epsilon > EPS_MIN:
            epsilon *= EPS_DECAY

        rewards.append(total_reward)
        print(f"Episode {episode+1}: Reward = {total_reward}, Epsilon = {epsilon:.3f}")

        # Alle 50 Episoden einmal rendern
        if (episode + 1) % 50 == 0:
            render_env = gym.make(env_name, render_mode="human")
            state, _ = render_env.reset()
            done = False
            while not done:
                with torch.no_grad():
                    action = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device)).argmax().item()
                state, _, terminated, truncated, _ = render_env.step(action)
                done = terminated or truncated
            render_env.close()

    env.close()

    # --- Plot Rewards ---
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN CartPole Training")
    plt.show()

    return policy_net, rewards

if __name__ == "__main__":
    trained_model, rewards = dqn_train()
