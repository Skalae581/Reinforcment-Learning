# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 10:46:01 2025

@author: TAKO
"""

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Füge diesen Fix am Anfang des Skripts hinzu
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
  




render_mode = "human" # or "rgb_array" for non-visualization
# render_mode = None # invisible mode, no rendering, MUCH faster training!
env = gym.make("CartPole-v1", render_mode=render_mode)
done = False
while not done:
    env.render()  # Hier muss render() aufgerufen werden
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)



# --- Policy-Netz ---
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

# --- Policy Gradient Baseline ---
def reinforce_cartpole(episodes=500, gamma=0.99, lr=0.01):
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for ep in range(episodes):
        states, actions, rewards = [], [], []
        state = env.reset()[0]

        # 1. Trajektorie sammeln
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor)
            action = np.random.choice(action_dim, p=probs.detach().numpy()[0])

            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # 2. Discounted Rewards berechnen
        discounted_rewards = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            discounted_rewards.insert(0, G)
        discounted_rewards = torch.FloatTensor(discounted_rewards)

        # Normalisieren für Stabilität
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        # 3. Policy Gradient Update
        optimizer.zero_grad()
        loss = 0
        for s, a, Gt in zip(states, actions, discounted_rewards):
            s_tensor = torch.FloatTensor(s).unsqueeze(0)
            probs = policy(s_tensor)
            log_prob = torch.log(probs[0, a])
            loss += -log_prob * Gt  # Gradient Ascent → Minimierung des negativen Erwartungswerts
        loss.backward()
        optimizer.step()

        # --- Monitoring ---
        total_reward = sum(rewards)
        print(f"Episode {ep+1}: Return = {total_reward}")

    env.close()

# Training starten
if __name__ == "__main__":
    reinforce_cartpole()
