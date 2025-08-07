# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 12:59:13 2025

@author: TAKO
"""

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

episodes=500
gamma=0.99
lr=3e-3

# render_mode = "human"
render_mode = None
env = gym.make('CartPole-v1', render_mode=render_mode)  # pole with visualization


# DDPG logic

# Actor Network (Policy)
observation_size = 4  # pole state: position, velocity, angle, angular-velocity
hidden_size = 128

class Actor(nn.Module):
	def __init__(self, obs_size, hidden_size, action_size):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, action_size),
			nn.Tanh() # continuous action space, output in range [-1, 1]
		)

	def forward(self, x):
		return self.model(x)

# Critic Network (Q-value)
class Critic(nn.Module):
	def __init__(self, obs_size, action_size, hidden_size):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(obs_size + action_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1) # quality of action
		)

	def forward(self, state, action):
		x = torch.cat([state, action], dim=-1)
		return self.model(x)

# Hyperparameters
tau = 0.005
buffer_limit = 100000
batch_size = 64
def print_model_summary(model, name="Model"):
    print(f"🧠 {name} Summary")
    print("=" * 50)
    total_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            shape = tuple(p.shape)
            count = p.numel()
            total_params += count
            print(f"{n:<30} | shape: {str(shape):<20} | params: {count}")
    print("=" * 50)
    print(f"🔢 Total trainable parameters: {total_params}")
    print("=" * 50)

# Initialize actor and critic
actor = Actor(observation_size, hidden_size, 1)
critic = Critic(observation_size, 1, hidden_size)
# load pretrained weights if available
try:
		# pass
		actor.load_state_dict(torch.load("actor.pth", weights_only=True))
		critic.load_state_dict(torch.load("critic.pth", weights_only=True))
		print("Loaded pretrained actor and critic models")
except:
		print("No pretrained models found, starting training from scratch")

target_actor = Actor(observation_size, hidden_size, 1)
target_critic = Critic(observation_size, 1, hidden_size)
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())
print_model_summary(actor, "Actor")
print_model_summary(critic, "Critic")

actor_optim = optim.Adam(actor.parameters(), lr=1e-4)
critic_optim = optim.Adam(critic.parameters(), lr=1e-3)

replay_buffer = deque(maxlen=buffer_limit)
reward_log = []

def soft_update(target, source):
	for t_param, s_param in zip(target.parameters(), source.parameters()):
		t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)

def sample_batch():
	batch = random.sample(replay_buffer, batch_size)
	state, action, reward, next_state, done = zip(*batch)
	return (torch.tensor(state, dtype=torch.float32),
			torch.tensor(action, dtype=torch.float32).unsqueeze(1),
			torch.tensor(reward, dtype=torch.float32).unsqueeze(1),
			torch.tensor(next_state, dtype=torch.float32),
			torch.tensor(done, dtype=torch.float32).unsqueeze(1))

def train():
	s_batch, a_batch, r_batch, ns_batch, d_batch = sample_batch()

	# Critic update
	with torch.no_grad():
		target_q = r_batch + gamma * target_critic(ns_batch, target_actor(ns_batch)) * (1 - d_batch)
	q_val = critic(s_batch, a_batch)
	critic_loss = nn.MSELoss()(q_val, target_q)
	critic_optim.zero_grad()
	critic_loss.backward()
	critic_optim.step()

	# Actor update
	actor_loss = -critic(s_batch, actor(s_batch)).mean()
	actor_optim.zero_grad()
	actor_loss.backward()
	actor_optim.step()

	# Soft update
	soft_update(target_actor, actor)
	soft_update(target_critic, critic)


for episode in range(episodes):
	state = env.reset()[0]
	total_reward = 0
	done = False

	while not done:
		state_tensor = torch.tensor(state, dtype=torch.float32)
		with torch.no_grad():
			action = actor(state_tensor).item()
			action = np.clip(action, -1, 1)
			action_discrete = 0 if action < 0 else 1 # binärer schalter

		next_state, reward, terminated, truncated, _ = env.step(action_discrete)
		done = terminated or truncated
		replay_buffer.append((state, action, reward, next_state, float(done)))
		state = next_state
		total_reward += reward

		if len(replay_buffer) < batch_size:
			continue
		train()
	reward_log.append(total_reward)

	if episode % 5 == 0:
		print(f"Episode {episode }, Total Reward: {total_reward:.2f}")
		# save model weights
		torch.save(actor.state_dict(), 'actor.pth')
		torch.save(critic.state_dict(), 'critic.pth')
def print_replay_buffer(buffer, max_items=5):
    print("\n🎲 Replay Buffer Preview:")
    print("=" * 50)
    print(f"Total transitions stored: {len(buffer)}")
    for i, transition in enumerate(list(buffer)[-max_items:]):
        state, action, reward, next_state, done = transition
        print(f"{i+1:>2}) s: {state}, a: {action:.3f}, r: {reward}, s': {next_state}, done: {done}")
    print("=" * 50)
if episode % 5 == 0:
		print_replay_buffer(replay_buffer)

# plotting
import matplotlib.pyplot as plt
plt.plot(reward_log)
plt.title('DDPG Training Rewards')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
# close environment
env.close()