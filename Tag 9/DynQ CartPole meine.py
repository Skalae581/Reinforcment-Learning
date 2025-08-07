import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Diskretisierung (optional, kann für State-Normalisierung nützlich sein)
def normalize_state(obs):
    return np.array(obs, dtype=np.float32)

# Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.model(x)

# Dynamics Model: (s,a) → (s', r)
class DynamicsModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameter
episodes = 500
alpha = 0.01
gamma = 0.99
epsilon = 0.1
planning_steps = 10
replay_buffer = deque(maxlen=10000)
batch_size = 64

# Umgebung
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Modelle
q_net = QNetwork(state_size, action_size)
q_optimizer = optim.Adam(q_net.parameters(), lr=alpha)
q_loss_fn = nn.MSELoss()

dynamics_model = DynamicsModel(state_size + 1, 128, state_size + 1)
dyn_optimizer = optim.Adam(dynamics_model.parameters(), lr=0.001)
dyn_loss_fn = nn.MSELoss()

reward_log = []

for ep in range(episodes):
    obs, _ = env.reset()
    state = normalize_state(obs)
    total_reward = 0
    done = False

    while not done:
        # ε-greedy
        if random.random() < epsilon:
            action = random.choice([0, 1])
        else:
            with torch.no_grad():
                q_values = q_net(torch.tensor(state).unsqueeze(0))
                action = torch.argmax(q_values).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = normalize_state(next_obs)

        # Save in replay buffer
        replay_buffer.append((state, action, reward, next_state, done))

        # Q-Learning Update (echte Erfahrung)
        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states_tensor = torch.tensor(states)
            actions_tensor = torch.tensor(actions).unsqueeze(1)
            rewards_tensor = torch.tensor(rewards).unsqueeze(1)
            next_states_tensor = torch.tensor(next_states)
            dones_tensor = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

            q_values = q_net(states_tensor).gather(1, actions_tensor)
            with torch.no_grad():
                next_q_values = q_net(next_states_tensor).max(1, keepdim=True)[0]
                targets = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

            loss = q_loss_fn(q_values, targets)
            q_optimizer.zero_grad()
            loss.backward()
            q_optimizer.step()

        # Trainiere Dynamikmodell
        if len(replay_buffer) >= batch_size:
            model_batch = random.sample(replay_buffer, batch_size)
            m_states, m_actions, m_rewards, m_next_states, _ = zip(*model_batch)

            x = torch.tensor([np.concatenate([s, [a]]) for s, a in zip(m_states, m_actions)], dtype=torch.float32)
            y = torch.tensor([np.concatenate([s_next, [r]]) for s_next, r in zip(m_next_states, m_rewards)], dtype=torch.float32)

            y_pred = dynamics_model(x)
            dyn_loss = dyn_loss_fn(y_pred, y)
            dyn_optimizer.zero_grad()
            dyn_loss.backward()
            dyn_optimizer.step()

        # Planning-Schritte (modellbasiert)
        for _ in range(planning_steps):
            if len(replay_buffer) < batch_size:
                break
            s, a, _, _, _ = random.choice(replay_buffer)
            s_tensor = torch.tensor(np.concatenate([s, [a]]), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred = dynamics_model(s_tensor).squeeze(0).numpy()
            s_next_pred = pred[:-1]
            r_pred = pred[-1]
            done_pred = False  # Wir schätzen `done` hier nicht – vereinfacht

            # Q-Update mit Modell
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            a_tensor = torch.tensor([[a]])
            r_tensor = torch.tensor([[r_pred]])
            s_next_tensor = torch.tensor(s_next_pred, dtype=torch.float32).unsqueeze(0)
            done_tensor = torch.tensor([[0.0]])

            q_val = q_net(s_tensor).gather(1, a_tensor)
            with torch.no_grad():
                q_next = q_net(s_next_tensor).max(1, keepdim=True)[0]
                q_target = r_tensor + gamma * q_next * (1 - done_tensor)

            q_loss = q_loss_fn(q_val, q_target)
            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

        state = next_state
        total_reward += reward

    reward_log.append(total_reward)
    if ep % 10 == 0:
        print(f"Episode {ep}, Reward: {total_reward:.2f}")

env.close()

# Plot
plt.plot(reward_log)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Deep Dyna-Q on CartPole")
plt.grid()
plt.show()
