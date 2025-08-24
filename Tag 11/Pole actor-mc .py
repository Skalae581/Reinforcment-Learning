# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 11:41:10 2025

@author: TAKO
"""

# import gym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

nr_training_steps = 1000
gamma = 0.99
lr = 1e-3

# Gym Environment
env = gym.make('CartPole-v1', render_mode="human")  # pole with visualization
# env = gym.make('CartPole-v1')  # pole without visualization

# Policy Network
observation_size = 4  # pole state: (position, velocity, angle, angular-velocity)
hidden_size = 128
n_actions = 2  # left / right
policy = nn.Sequential(  # batch wird bei pytorch IMPLIZIT mir durchgeschliffen!
	nn.Linear(observation_size, hidden_size),  # keine batch size mitgeben
	nn.ReLU(),
	nn.Linear(hidden_size, n_actions),
	nn.Softmax(dim=-1)
)
optimizer = optim.Adam(policy.parameters(), lr=lr)

value_function = nn.Sequential(
	nn.Linear(observation_size, hidden_size),
	nn.ReLU(),
	nn.Linear(hidden_size, 1)
)
value_optimizer = optim.Adam(value_function.parameters(), lr=lr)

# load weights if available
try:
	policy.load_state_dict(torch.load("cartpole_ac_policy.pth", weights_only=True))
	value_function.load_state_dict(torch.load("cartpole_ac_value.pth", weights_only=True))
except:
	print("No weights available, starting from scratch.")
def print_model_summary(model):
    total_params = 0
    print("🧠 Model Summary:")
    print("=" * 50)
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape = tuple(param.shape)
            n_params = param.numel()
            total_params += n_params
            print(f"{name:<30} | shape: {str(shape):<15} | params: {n_params}")
    print("=" * 50)
    print(f"🔢 Total trainable parameters: {total_params}")
    print("=" * 50)


run_lengths = []
for training_step in range(nr_training_steps):
	state = env.reset()[0]
	log_probs = []
	values = []
	rewards = []

	done = False
	steps = 0
	while not done:
		steps += 1
		state_tensor = torch.tensor(state, dtype=torch.float32)
		probs = policy(state_tensor)
		value = value_function(state_tensor)

		action = torch.multinomial(probs, num_samples=1).item()
		log_prob = torch.log(probs[action])

		log_probs.append(log_prob)
		values.append(value)

		state, reward, terminated, truncated, info = env.step(action)
		rewards.append(reward)
		done = terminated or truncated

	run_lengths.append(steps)

	# Compute returns and advantages
	# 15 Zeilen Code für Monte Carlo Returns, immer noch überschaubar!
	gains = []
	G = 0  # zusammen-zählen der Belohnungen als Gewinn
	for r in reversed(rewards):
		G = r + gamma * G
		gains.insert(0, G)
	gains = torch.tensor(gains, dtype=torch.float32)  # Gesamtgewinn
	values = torch.stack(values).squeeze()
	with torch.no_grad():
		advantages = gains - values#.detach()  # Vorteile = Gewinne - geschätzter Werte des Zustands

	# Normalize advantages (optional)
	advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

	log_probs = torch.stack(log_probs)
	policy_loss = -(log_probs * advantages).sum()
	value_loss = nn.functional.mse_loss(values, gains)

	optimizer.zero_grad()
	policy_loss.backward()
	optimizer.step()

	value_optimizer.zero_grad()
	value_loss.backward()
	value_optimizer.step()

	if training_step % 50 == 0:
           print_model_summary(value_function)

           print("=" * 60)
           print(f"📘 Episode             : {training_step}")
           print(f"📈 Steps in Episode    : {steps}")
           print(f"🧮 Policy Loss         : {policy_loss.item():.6f}")
           print(f"💰 Value Loss          : {value_loss.item():.6f}")
           
           print("=" * 60)
        
           plt.figure(figsize=(10, 4))
           plt.plot(run_lengths, label="Episodenlänge")
           #plt.axhline(y=avg_run_length, color='red', linestyle='--', label="Durchschnitt")
           plt.title("📉 Trainingsverlauf - Episodenlänge")
           plt.xlabel("Episode")
           plt.ylabel("Schritte bis Abbruch")
           plt.grid(True)
           plt.legend()
           plt.tight_layout()
           plt.savefig("run_lengths.png")
           plt.show()
           plt.pause(0.001)
           plt.close()
def log_training_summary(training_step, steps, policy_loss, value_loss, value_function, policy=None):
    print("=" * 60)
    print(f"📘 Episode             : {training_step}")
    print(f"📈 Steps in Episode    : {steps}")
    print(f"🧮 Policy Loss         : {policy_loss.item():.6f}")
    print(f"💰 Value Loss          : {value_loss.item():.6f}")
    print("=" * 60)

    def print_model_summary(model, title):
        print(f"🧠 {title} Summary:")
        print("=" * 60)
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                shape = tuple(param.shape)
                n_params = param.numel()
                total_params += n_params
                print(f"{name:<30} | shape: {str(shape):<15} | params: {n_params}")
        print("=" * 60)
        print(f"🔢 Total trainable parameters: {total_params}")
        print("=" * 60)

    print_model_summary(value_function, "Critic (Value Function)")

    if policy is not None:
        print_model_summary(policy, "Actor (Policy)")
           
# save the policy and valuefunction
torch.save(policy.state_dict(), "cartpole_ac_policy.pth")
torch.save(value_function.state_dict(), "cartpole_ac_value.pth")
print_model_summary(policy)

# close the environment
env.close()