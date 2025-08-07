import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# One-hot encoding for discrete states
def one_hot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec

# Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

# Critic Network (State-Value Function)
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
gamma = 0.99
episodes = 1000
learning_rate = 0.01
max_steps = 100
hidden_size = 128

# Environment
env = gym.make("FrozenLake-v1", is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n

# Initialize networks and optimizers
actor = Actor(state_size, hidden_size, action_size)
critic = Critic(state_size, hidden_size)
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)

# Print model summaries
print("\nActor Network:")
print(actor)
print("\nCritic Network:")
print(critic)

# === Trainingsmetriken ===
successes = []  # 1 = Ziel erreicht, 0 = nicht
total_rewards = []

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    log_probs = []
    values = []
    rewards = []
    total_reward = 0

    for step in range(max_steps):
        state_vec = torch.tensor(one_hot(state, state_size)).unsqueeze(0)
        probs = actor(state_vec)
        value = critic(state_vec)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        print(f"Episode {episode}, Step {step}: State {state}, Action {action.item()}, Probabilities {probs.detach().numpy()}")

        next_state, reward, done, _, _ = env.step(action.item())

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        total_reward += reward

        state = next_state

        if done:
            break

    # Erfolg = Ziel erreicht
    successes.append(1 if reward > 0 else 0)
    total_rewards.append(total_reward)

    # Compute returns and advantages
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    values = torch.cat(values).squeeze()
    if returns.std() > 0:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    advantages = returns - values.detach()

    # Actor loss (policy gradient with advantage)
    actor_loss = torch.stack([-log_prob * adv for log_prob, adv in zip(log_probs, advantages)]).sum()

    # Critic loss (value approximation)
    critic_loss = nn.functional.mse_loss(values, returns)

    # Optimize actor
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # Optimize critic
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Success Rate (last 100): {np.mean(successes[-100:])*100:.1f}%")

# === PLOTTEN: Zielerreichung ===
plt.figure(figsize=(10, 4))

# 1. Binärer Ziel-Erreichungsplot
plt.subplot(1, 2, 1)
plt.plot(successes, label="Ziel erreicht (1 = Ja, 0 = Nein)")
plt.xlabel("Episode")
plt.ylabel("Ziel erreicht?")
plt.title("Zielerreichung pro Episode")
plt.grid(True)
plt.legend()

# 2. Gleitender Durchschnitt der Erfolgsrate
plt.subplot(1, 2, 2)
window = 20
if len(successes) >= window:
    rolling_success = np.convolve(successes, np.ones(window)/window, mode='valid')
    plt.plot(rolling_success, label=f"Erfolgsrate Ø über {window} Episoden")
    plt.xlabel("Episode")
    plt.ylabel("Erfolgsrate")
    plt.title("Erfolgsrate über Zeit")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
