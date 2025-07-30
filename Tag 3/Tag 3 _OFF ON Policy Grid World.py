# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 15:04:44 2025

@author: TAKO
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# Grid-Einstellungen
GRID_SIZE = 4
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_SYMBOLS = ['⬆', '⬇', '⬅', '➡']
ACTION_EFFECTS = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

# Ziel
TERMINAL_STATE = (0, 0)
REWARD_GOAL = 10
REWARD_STEP = -1
EPSILON = 0.1
ALPHA = 0.5
GAMMA = 0.9
EPISODES = 3000

def is_valid(state):
    x, y = state
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def next_state(state, action):
    dx, dy = ACTION_EFFECTS[action]
    new_state = (state[0] + dx, state[1] + dy)
    return new_state if is_valid(new_state) else state

def epsilon_greedy(q, state):
    if random.random() < EPSILON:
        return random.randint(0, 3)
    return np.argmax(q[state])

def sarsa(q):
    for _ in range(EPISODES):
        state = (GRID_SIZE - 1, GRID_SIZE - 1)
        action = epsilon_greedy(q, state)
        while state != TERMINAL_STATE:
            next_s = next_state(state, action)
            reward = REWARD_GOAL if next_s == TERMINAL_STATE else REWARD_STEP
            next_a = epsilon_greedy(q, next_s)
            td_target = reward + GAMMA * q[next_s][next_a]
            q[state][action] += ALPHA * (td_target - q[state][action])
            state, action = next_s, next_a

def q_learning(q):
    for _ in range(EPISODES):
        state = (GRID_SIZE - 1, GRID_SIZE - 1)
        while state != TERMINAL_STATE:
            action = epsilon_greedy(q, state)
            next_s = next_state(state, action)
            reward = REWARD_GOAL if next_s == TERMINAL_STATE else REWARD_STEP
            best_next = np.max(q[next_s])
            td_target = reward + GAMMA * best_next
            q[state][action] += ALPHA * (td_target - q[state][action])
            state = next_s

def visualize(q, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            state = (x, y)
            if state == TERMINAL_STATE:
                ax.add_patch(plt.Rectangle((y, x), 1, 1, facecolor='green', edgecolor='black'))
                ax.text(y + 0.5, x + 0.5, "GOAL", ha='center', va='center', fontsize=10, color='white')
            else:
                best_a = np.argmax(q[state])
                value = np.max(q[state])
                ax.add_patch(plt.Rectangle((y, x), 1, 1, facecolor='white', edgecolor='black'))
                ax.text(y + 0.5, x + 0.5, f"{value:.1f}\n{ACTION_SYMBOLS[best_a]}",
                        ha='center', va='center', fontsize=9, color='blue')

    ax.set_xticks(np.arange(GRID_SIZE + 1))
    ax.set_yticks(np.arange(GRID_SIZE + 1))
    ax.grid(True)
    ax.set_title(title)
    plt.gca().invert_yaxis()
    plt.show()

# On-Policy: SARSA
q_sarsa = { (x, y): np.zeros(4) for x in range(GRID_SIZE) for y in range(GRID_SIZE) }
sarsa(q_sarsa)
visualize(q_sarsa, "On-Policy (SARSA) mit Bootstrapping")

# Off-Policy: Q-Learning
q_qlearn = { (x, y): np.zeros(4) for x in range(GRID_SIZE) for y in range(GRID_SIZE) }
q_learning(q_qlearn)
visualize(q_qlearn, "Off-Policy (Q-Learning) mit Bootstrapping")
