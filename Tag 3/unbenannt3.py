# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 10:51:32 2025

@author: TAKO
"""

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Grid dimensions
GRID_SIZE = 4
TERMINAL_STATE = (0, 0)
DISCOUNT_FACTOR = .99 # bei 1 läuft der Algorithmus im Kreis: nicht konvergierend
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
		'right':  '➡'
}

# Reward matrix
rewards = np.full((GRID_SIZE, GRID_SIZE), -1.0) # -1 pro Schritt
rewards[:-1, 1] = -10  # 🟥 Red wall cells
# rewards[2, 3] = -10
# rewards[4, 4] = -10
rewards[3, 3] = -10
rewards[3,1] = -10
rewards[TERMINAL_STATE] = 100
#rewards[:-1, 1] = -2
#rewards[3,1] = -2
#rewards[3,3] = -2

# Initialize value function and policy
value_function = np.zeros((GRID_SIZE, GRID_SIZE)) # boot strap with zeros
policy = np.full((GRID_SIZE, GRID_SIZE), 'up', dtype=object)

# Check if a state is within bounds
def is_valid_state(state):
		x, y = state
		return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE
# Policy evaluation : UPDATE the value function for each state based on the current policy
# Breche ab wenn die Änderung der Wertefunktion kleiner als theta ist (Konvergenz der Wertefunktion)
def evaluate_policy(value_function, policy, theta=1e-4):
		while True:
				delta = 0
				new_value_function = np.copy(value_function)
				for x in range(GRID_SIZE):
						for y in range(GRID_SIZE):
								state = (x, y)
								if state == TERMINAL_STATE:
										continue
								action = policy[x, y]
								dx, dy = ACTION_EFFECTS[action]
								next_state = (x + dx, y + dy) if is_valid_state((x + dx, y + dy)) else (x, y)
								# is_valid_state reflects the 'probability' of the action
								new_value = rewards[state] + DISCOUNT_FACTOR * value_function[next_state] # Bellman equation term!
								new_value_function[state] = new_value
								delta = max(delta, abs(new_value - value_function[state]))
				value_function[:] = new_value_function
				if delta < theta:
						break

# Policy improvement
# Update the policy based on the current value function!
def improve_policy(value_function, policy):
		policy_stable = True
		# π(s) = argmaxₐ R(s,a) + γ * V(s')
		for x in range(GRID_SIZE):
				for y in range(GRID_SIZE):
						state = (x, y)
						if state == TERMINAL_STATE:
								continue
						action_values = {}
						for action, (dx, dy) in ACTION_EFFECTS.items():
								next_state = (x + dx, y + dy) if is_valid_state((x + dx, y + dy)) else (x, y)
								action_values[action] = rewards[state] + DISCOUNT_FACTOR * value_function[next_state]
						best_action = max(action_values, key=action_values.get)
						if policy[x, y] != best_action:
								policy_stable = False # höre erst auf wenn policy sich nicht mehr ändert
						policy[x, y] = best_action
		return policy_stable

# Value Iteration Loop
# Abwechselnd Policy Evaluation (Werte Schätzung verbessern) und Policy Improvement (Policy verbessern)
def policy_iteration(value_function, policy):
		iteration = 0
		while True:
				evaluate_policy(value_function, policy)
				policy_stable = improve_policy(value_function, policy)
				iteration += 1
				visualize_policy_and_values(value_function, policy, iteration)
				if policy_stable:
						break

# Visualization
def visualize_policy_and_values(value_function, policy, iteration):
	fig, ax = plt.subplots(figsize=(8, 8))
	for x in range(GRID_SIZE):
		for y in range(GRID_SIZE):
			if (x, y) == TERMINAL_STATE:  # Terminal state
				rect = plt.Rectangle((y - 0.5, x - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='grey')
			elif rewards[x, y] == -10:  # -10 states
				rect = plt.Rectangle((y - 0.5, x - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='red')
			else:  # Other states
				rect = plt.Rectangle((y - 0.5, x - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='white')

			ax.add_patch(rect)  # Adding the rectangle before the text
			ax.text(y, x, f"{value_function[x, y]:.1f}\n{ACTION_SYMBOLS[policy[x, y]]}",
							ha='center', va='center', fontsize=8, color='blue')

	ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1))
	ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.grid(color='black')

	ax.set_title(f"Iteration {iteration}")
	plt.gca().invert_yaxis()
	plt.show()

# Run the algorithm
policy_iteration(value_function, policy)