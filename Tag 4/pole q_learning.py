# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 11:11:43 2025

@author: TAKO
"""

import gymnasium as gym
import numpy as np
import pickle

# Discretization parameters
n_bins = (6, 12, 6, 12)  # bins per state dimension
lower_bounds = np.array([-4.8, -5.0, -0.418, -5.0])
upper_bounds = np.array([4.8, 5.0, 0.418, 5.0])

# Environment setup
render_mode = "human"  # or "rgb_array" for non-visualization
# render_mode = None  # invisible mode, no rendering, MUCH faster training!
env = gym.make("CartPole-v1", render_mode=render_mode)

save_file = "pole-baselines.pkl"


# Q-table initialization
q_table = np.zeros(n_bins + (2,))  # 2 actions: left (0), right (1)

# load existing Q-table if available
try:
    q_table = pickle.load(open(save_file, "rb"))
except FileNotFoundError:
  print("Kann keinen Q-Table nicht laden, starte mit leerem Q-Table.")



def discretize(state):
    ratios = (state - lower_bounds) / (upper_bounds - lower_bounds)
    bins = [min(n - 1, max(0, int(r * n))) for r, n in zip(ratios, n_bins)]
    return tuple(bins)

# Hyperparameters
alpha = 0.01    # learning rate
gamma = 0.99   # discount factor
epsilon = 1.0  # initial exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
episodes = 1000
max_steps = 500
if episodes % 100 == 0:
    import pickle
    pickle.dump(q_table, open(save_file, "wb"))
#print("Q-Table Shape:", q_table.shape)
#print(q_table)

def policy(state):
  global epsilon
  if np.random.rand() < epsilon:
    action = env.action_space.sample()
  else:
    action = np.argmax(q_table[state])
  return action

def q_update(state, action, next_state, reward, done):
  best_next_action = np.argmax(q_table[next_state])
  best_next_reward = q_table[next_state][best_next_action]
  # best_next_reward = max(q_table[next_state])
  td_target = reward + gamma * best_next_reward
  old_reward_estimate = q_table[state][action]
  q_table[state][action] += alpha * (td_target - old_reward_estimate)


for episode in range(episodes):
    obs, _ = env.reset()
    state = discretize(obs)

    for step in range(max_steps):
        action = policy(state)  # use epsilon-greedy policy

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize(next_obs)

        # Q-update
        q_update(state, action, next_state, reward, done)
        # Ausgabe der aktualisierten Stelle
        print(f"Q[{state}][{action}] = {q_table[state][action]:.4f}")
       # print(q_table)
        state = next_state
        if done:
            break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

env.close()
# SAVE q_table
import pickle

pickle.dump(q_table, open(save_file, "wb"))
#print("Q-Table Shape:", q_table.shape)
#print(q_table)
# print("Training completed in %.2f seconds." % (time.time() - start_time))