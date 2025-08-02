# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 12:09:44 2025

@author: TAKO
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# --- Grundkonfiguration ---
GRID_SIZE = 4
TERMINAL_STATE = (0, 0)
DISCOUNT_FACTOR = 0.99
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_EFFECTS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}
ACTION_SYMBOLS = {
    'up': '⬆',
    'down': '⬇',
    'left': '⬅',
    'right': '➡'
}

# Rewards
rewards = np.full((GRID_SIZE, GRID_SIZE), -1.0)  # -1 pro Schritt
rewards[:-1, 1] = -10  # Hindernisse
rewards[3, 3] = -10

# Hyperparameter
ALPHA = 0.1
EPSILON = 0.1
EPISODES = 5000

# Hilfsfunktionen
def is_terminal(state):
    return state == TERMINAL_STATE

def step(state, action):
    dy, dx = ACTION_EFFECTS[action]
    y, x = state
    ny, nx = y + dy, x + dx
    if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
        next_state = (ny, nx)
    else:
        next_state = state
    reward = rewards[next_state]
    return next_state, reward

def epsilon_greedy(Q, state):
    if random.random() < EPSILON:
        return random.choice(range(len(ACTIONS)))
    else:
        y, x = state
        return np.argmax(Q[y, x])

# --- SARSA Algorithmus ---
Q_sarsa = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

for episode in range(EPISODES):
    state = (GRID_SIZE - 1, GRID_SIZE - 1)  # Start unten rechts
    action_idx = epsilon_greedy(Q_sarsa, state)

    while not is_terminal(state):
        action = ACTIONS[action_idx]
        next_state, reward = step(state, action)

        # Nächste Aktion (On-Policy)
        next_action_idx = epsilon_greedy(Q_sarsa, next_state)

        # SARSA-Update
        y, x = state
        ny, nx = next_state
        Q_sarsa[y, x, action_idx] += ALPHA * (
            reward + DISCOUNT_FACTOR * Q_sarsa[ny, nx, next_action_idx] - Q_sarsa[y, x, action_idx]
        )

        state = next_state
        action_idx = next_action_idx

# --- Policy extrahieren ---
policy_sarsa = np.full((GRID_SIZE, GRID_SIZE), '', dtype=object)
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        if is_terminal((y, x)):
            policy_sarsa[y, x] = '🟩'
        else:
            best_action = np.argmax(Q_sarsa[y, x])
            policy_sarsa[y, x] = ACTION_SYMBOLS[ACTIONS[best_action]]

# --- Policy anzeigen ---
print("Policy (SARSA):")
for row in policy_sarsa:
    print(' '.join(row))

# --- Heatmaps der Q-Werte ---
for i, action in enumerate(ACTIONS):
    plt.figure()
    plt.title(f"Q-Werte (SARSA) für Aktion: {action}")
    plt.imshow(Q_sarsa[:, :, i], cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.show()
