# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:25:40 2025

@author: TAKO
"""

import random

import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch

WIDTH = 12
HEIGHT = 4

# https://www.gymlibrary.dev/environments/toy_text/frozen_lake/
# render_mode="ansi"  # "rgb_array" for image array, "ansi" for text output
render_mode="human"  #  visualization
env = gym.make('CliffWalking-v1',render_mode=render_mode)

TERMINAL_STATE = (WIDTH - 1, HEIGHT - 1)  # Bottom right corner is the goal

max_episodes = 5000
DISCOUNT_FACTOR = .99
exploration_steps = 100
epsilon = 0.9  # Exploration rate for epsilon-greedy policy

ACTION_EFFECTS = {
	0: (-1, 0),  # UP
	1: (0, 1),  # RIGHT
	2: (1, 0),  # DOWN
	3: (0, -1),  # LEFT
}
ACTION_SYMBOLS = {
	0: '⬆',
	1: '➡',
	2: '⬇',
	3: '⬅'
}
# Reward matrix FILL FROM ENVIRONMENT!!
rewards = np.full((WIDTH, HEIGHT), .0)  # 0 pro Schritt
value_function = np.zeros((WIDTH, HEIGHT))  # boot strap with zeros
policy_matrix = np.zeros((WIDTH, HEIGHT))
success_path = np.zeros((WIDTH, HEIGHT)) # save successful actions as path
actions_tried = np.zeros((WIDTH, HEIGHT, 4))  # Count of action tried in each state

rewards[TERMINAL_STATE]=1000

# Check if a state is within bounds
def is_valid_state(state):
	x, y = state
	return 0 <= x < WIDTH and 0 <= y < HEIGHT


# Policy evaluation : UPDATE the value function for each state based on the current policy
# Breche ab wenn die Änderung der Wertefunktion kleiner als theta ist (Konvergenz der Wertefunktion)
def evaluate_policy(value_function, policy_matrix, theta=1e-4):
	while True:
		delta = 0
		new_value_function = np.copy(value_function)
		for x in range(WIDTH):
			for y in range(HEIGHT):
				state = (x, y)
				if state == TERMINAL_STATE:
					continue
				action = policy_matrix[x, y]
				dx, dy = ACTION_EFFECTS[action]
				next_state = (x + dx, y + dy) if is_valid_state((x + dx, y + dy)) else (x, y)
				# is_valid_state reflects the 'probability' of the action
				new_value = rewards[state] + DISCOUNT_FACTOR * value_function[next_state]  # Bellman equation term!
				new_value_function[state] = new_value
				delta = max(delta, abs(new_value - value_function[state]))
		value_function[:] = new_value_function
		if delta < theta:
			break


# Policy improvement
# Update the policy based on the current value function!
def improve_policy(value_function, policy_matrix):
	policy_stable = True
	# π(s) = argmaxₐ R(s,a) + γ * V(s')
	for x in range(WIDTH):
		for y in range(HEIGHT):
			state = (x, y)
			if state == TERMINAL_STATE:
				continue
			action_values = {}
			for action, (dx, dy) in ACTION_EFFECTS.items():
				next_state = (x + dx, y + dy) if is_valid_state((x + dx, y + dy)) else (x, y)
				action_values[action] = rewards[state] + DISCOUNT_FACTOR * value_function[next_state]
			best_action = max(action_values, key=action_values.get)
			if policy_matrix[x, y] != best_action:
				policy_stable = False  # höre erst auf wenn policy sich nicht mehr ändert
			policy_matrix[x, y] = best_action
	return policy_stable


def policy_random():
	return env.action_space.sample()  # Random action

# Aufgabe: schreibt eine eigen Policy
def policy_our(x,y):
	return 0

def policy_try_new(x,y):
	min = 2**31
	best_action = 0
	tries = 0
	for action in ACTION_SYMBOLS.keys():
		tries = actions_tried[x, y, action]
		if tries < min:
			min = tries
			best_action = action
	# print("best_action", best_action, ACTION_SYMBOLS[best_action], tries)
	return best_action

def policy_success(x,y):
	return int(success_path[x,y])

def policy(x, y, iteration):
	# if step < exploration_steps and random.random() < epsilon:
	# 	action = policy_random()
	# else:
	if iteration < exploration_steps:
		# action = policy_our(x,y)
		action = policy_random()
		# action = policy_try_new(x,y)
	else:
		# action = policy_success(x, y) # just follow best path
		action = int(policy_matrix[x, y])
	print(exploration_steps, iteration, "best_action", action, ACTION_SYMBOLS[action], actions_tried[x, y, action or 0])
	actions_tried[x,y,action]+=1
	success_path[x,y]=action
	return action

def train():
	global exploration_steps
	for step in range(max_episodes):
		state = torch.tensor(env.reset()[0])
		done = False
		reward = 0
		while not done:
			x = state.item() % WIDTH
			y = int(state.item() / WIDTH)
			rewards[x, y] = reward
			action = policy(x,y,step)
			next_state, reward, terminated, truncated, info = env.step(action)
			if terminated: reward = 100 # HACK: reward for reaching the goal
			# print(terminated, truncated, reward)
			if reward >= 0 or terminated:
				print("GOT NO PUNISHMENT (YAY!)", reward)
				exploration_steps = 0  # stop exploring
				# exit()
			state = torch.tensor(next_state)
			done = terminated or truncated or reward==-100  # pole is down or time is up!
			# BUG in env: NOT done when reaching the cliff!

		# if step % 10 == 0 :
		visualize_policy_and_values(value_function, policy_matrix,step)
		# if step > exploration_steps:  # sammle erst ein paar Daten zu rewards!
		if exploration_steps == 0:
			policy_stable = False
			while not policy_stable:
				evaluate_policy(value_function, policy_matrix, theta=1e-4)
				policy_stable = improve_policy(value_function, policy_matrix)
				visualize_policy_and_values(value_function, policy_matrix, step)
			print("Policy stable after", step, "steps")
			# exit()
		else: # train ONCE!
			evaluate_policy(value_function, policy_matrix, theta=1e-4)
			policy_stable = improve_policy(value_function, policy_matrix)

# Visualization
def visualize_policy_and_values(value_function, policy, iteration):
	fig, ax = plt.subplots(figsize=(8, 8))
	for x in range(WIDTH):
		for y in range(HEIGHT):
			if (x, y) == TERMINAL_STATE:  # Terminal state
				rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='grey')
			elif rewards[x, y] < -1:
				rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='red')
			else:  # Other states
				rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, linewidth=1, edgecolor='black', facecolor='white')

			ax.add_patch(rect)  # Adding the rectangle before the text
			ax.text(x,y, f"{value_function[x, y]:.1f}\n{ACTION_SYMBOLS[policy[x, y]]}\n{rewards[x, y]:.1f}",
							ha='center', va='center', fontsize=8, color='blue')

	ax.set_xticks(np.arange(-0.5, WIDTH, 1))
	ax.set_yticks(np.arange(-0.5, HEIGHT, 1))
	ax.set_xticklabels([])
	ax.set_yticklabels([])
	ax.grid(color='black')

	ax.set_title(f"Iteration {iteration}")
	plt.gca().invert_yaxis()
	plt.show()

train()