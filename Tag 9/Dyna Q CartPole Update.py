# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 08:42:29 2025

@author: TAKO
"""

import gymnasium as gym
import numpy as np
from collections import defaultdict
import random

# Diskretisierung der Zustände
def discretize_state(obs, bins):
    upper_bounds = [2.4, 3.0, 0.21, 2.5]
    lower_bounds = [-2.4, -3.0, -0.21, -2.5]
    ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((bins - 1) * ratios[i])) for i in range(len(obs))]
    new_obs = [min(bins - 1, max(0, new_obs[i])) for i in range(len(obs))]
    return tuple(new_obs)

# Parameter
bins = 6  # Diskretisierung pro Dimension
alpha = 0.1
gamma = 0.99
epsilon = 0.1
planning_steps = 10
episodes = 300

env = gym.make('CartPole-v1')
q_table = defaultdict(lambda: np.zeros(env.action_space.n))
model = dict()

reward_log = []

for ep in range(episodes):
    obs, _ = env.reset()
    state = discretize_state(obs, bins)
    total_reward = 0
    done = False

    while not done:
        # ε-greedy Policy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_obs, bins)

        # Model-Free Q-Learning Update
        best_next = np.max(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * best_next - q_table[state][action])

        # Modell speichern
        model[(state, action)] = (next_state, reward)

        # Planning-Schritte (simulierte Updates)
        for _ in range(planning_steps):
            s, a = random.choice(list(model.keys()))
            s_prime, r = model[(s, a)]
            best_s_prime = np.max(q_table[s_prime])
            q_table[s][a] += alpha * (r + gamma * best_s_prime - q_table[s][a])

        state = next_state
        total_reward += reward

    reward_log.append(total_reward)
    if ep % 10 == 0:
        print(f"Episode {ep}, Total Reward: {total_reward}")

env.close()
import matplotlib.pyplot as plt
plt.plot(reward_log)
plt.title("Dyna-Q Training on CartPole")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()
