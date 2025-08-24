# -*- coding: utf-8 -*-
"""
RL^2 Actor-Critic auf zufälligen FrozenLake-Tasks (Meta-RL)
Speichert: best.pt, final.pt, metrics.csv, returns.png, steps_to_goal.png, Maps & Video
"""

import os, csv, math
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

# ========================= Lauf/Output =========================
RUN_NAME = f"rl2_frozenlake_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
OUT_DIR = os.path.join("runs", RUN_NAME)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "maps"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "videos"), exist_ok=True)

USE_TENSORBOARD = True
if USE_TENSORBOARD:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "tb"))

BEST_SCORE = -1e9  # Moving-Avg-Bestwert

def save_checkpoint(path, model, optimizer, extra=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "extra": extra or {}
    }
    torch.save(payload, path)

def maybe_save_best(avg_return_recent, model, optimizer):
    global BEST_SCORE
    if avg_return_recent > BEST_SCORE:
        BEST_SCORE = avg_return_recent
        save_checkpoint(os.path.join(OUT_DIR, "best.pt"), model, optimizer,
                        extra={"best_ma_return": float(avg_return_recent), "run": RUN_NAME})

# ========================= Hyperparams =========================
HIDDEN_SIZE = 64
LR = 1e-3
EPISODES_PER_TASK = 100     # Inner-Loop Episoden pro Task
TASKS = 150                 # Anzahl verschiedener Tasks (Maps)
STEPS = 50
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
GAMMA = 0.99
MAP_SIZE = 4                # 4, 6, 8 …
MAP_P = 0.85                # Wahrscheinlichkeit für "F" (frei)
SLIPPERY = False
MA_WINDOW_BEST = 50         # Fenster für "bestes Modell"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================= Utils =========================
def one_hot(idx, size, device=DEVICE):
    t = torch.zeros(size, dtype=torch.float32, device=device)
    t[idx] = 1.0
    return t

def compute_returns(rewards, gamma):
    R = 0.0
    out = []
    for r in reversed(rewards):
        R = r + gamma * R
        out.append(R)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32, device=DEVICE)

def moving_avg(x, k=20):
    x = np.array(x, dtype=float)
    w = np.ones(k) / max(1, k)
    y = np.convolve(np.nan_to_num(x), w, mode="valid")
    pad = np.full(k - 1, np.nan)
    return np.concatenate([pad, y])

# ========================= Modell =========================
class RL2ActorCritic(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size):
        super().__init__()
        input_dim = obs_size + action_size + 1  # obs_onehot + last_action_onehot + last_reward
        self.fc_in = nn.Linear(input_dim, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size)
        self.pi  = nn.Linear(hidden_size, action_size)
        self.v   = nn.Linear(hidden_size, 1)

    def forward(self, obs, last_action, last_reward, h):
        x = torch.cat([obs, last_action, last_reward], dim=-1)     # (1, input_dim)
        x = torch.relu(self.fc_in(x)).unsqueeze(0)                  # (seq=1,batch=1,hidden)
        out, h = self.rnn(x, h)                                     # (1,1,hidden)
        out = out.squeeze(0)                                        # (1, hidden)
        logits = self.pi(out)                                       # (1, A)
        value  = self.v(out).squeeze(-1)                            # (1,)
        return logits, value, h

# ========================= Task-Factory =========================
def make_task_env(task_id, size=MAP_SIZE, p=MAP_P, slippery=SLIPPERY):
    desc = generate_random_map(size=size, p=p)   # Liste von Strings
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=slippery)
    env.reset(seed=task_id)
    return env, desc

# Spaces robust bestimmen (für beliebige MAP_SIZE)
_probe_desc = generate_random_map(size=MAP_SIZE, p=MAP_P)
_probe_env  = gym.make("FrozenLake-v1", desc=_probe_desc, is_slippery=SLIPPERY)
obs_size = _probe_env.observation_space.n
action_size = _probe_env.action_space.n
_probe_env.close()

net = RL2ActorCritic(obs_size, action_size, HIDDEN_SIZE).to(DEVICE)
opt = optim.Adam(net.parameters(), lr=LR)

episode_returns, steps_to_goal = [], []

# ========================= Meta-Training =========================
for task in range(TASKS):
    env, desc = make_task_env(task)
    with open(os.path.join(OUT_DIR, "maps", f"task_{task:04d}.txt"), "w") as f:
        f.write("\n".join(desc))

    # Inner-Loop: Hidden-State über Episoden derselben Task behalten
    h = torch.zeros(1, 1, HIDDEN_SIZE, device=DEVICE)

    for ep in range(EPISODES_PER_TASK):
        obs, _ = env.reset()
        obs_oh = one_hot(obs, obs_size)
        last_action = torch.zeros(action_size, device=DEVICE)
        last_reward = torch.zeros(1, device=DEVICE)

        log_probs, values, rewards, entropies = [], [], [], []
        steps, reached = 0, False

        for t in range(STEPS):
            logits, value, h = net(
                obs_oh.unsqueeze(0),
                last_action.unsqueeze(0),
                last_reward.view(1, 1),
                h
            )
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            obs_next, reward, terminated, truncated, _ = env.step(action.item())

            # optional shaping:
            # reward = reward - 0.01

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(float(reward))
            entropies.append(entropy)

            steps += 1
            if reward == 1.0:
                reached = True

            obs_oh = one_hot(obs_next, obs_size)
            last_action = one_hot(action.item(), action_size)
            last_reward = torch.tensor([reward], dtype=torch.float32, device=DEVICE)

            if terminated or truncated:
                break

        # episodischer Update
        if len(rewards) > 0:
            R = compute_returns(rewards, GAMMA)
            V = torch.cat(values)                 # (T,)
            A = R - V.detach()                    # Advantage

            policy_loss  = -(torch.stack(log_probs) * A).mean()
            value_loss   = nn.functional.mse_loss(V, R)
            entropy_loss = -torch.stack(entropies).mean()

            loss = policy_loss + VALUE_COEF * value_loss + ENTROPY_COEF * entropy_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        # ganz wichtig: Graph trennen für nächste Episode derselben Task
        h = h.detach()

        # Logging
        ep_return = sum(rewards)
        episode_returns.append(ep_return)
        steps_to_goal.append(steps if reached else math.nan)

        if USE_TENSORBOARD:
            global_ep = len(episode_returns)
            writer.add_scalar("train/episodic_return", ep_return, global_ep)
            writer.add_scalar("train/episode_steps", steps, global_ep)

        # Bestes Modell (Moving-Avg über letzte MA_WINDOW_BEST Episoden)
        recent = [r for r in episode_returns[-MA_WINDOW_BEST:]]
        if len(recent) > 0:
            ma = float(np.nanmean(recent))
            maybe_save_best(ma, net, opt)

    env.close()

# ========================= Speichern: final, CSV, Plots =========================
# final checkpoint
save_checkpoint(os.path.join(OUT_DIR, "final.pt"), net, opt,
                extra={"run": RUN_NAME, "episodes": len(episode_returns)})

# CSV
csv_path = os.path.join(OUT_DIR, "metrics.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["episode", "return", "steps_to_goal"])
    for i, (ret, steps) in enumerate(zip(episode_returns, steps_to_goal), start=1):
        w.writerow([i, ret, ("" if (steps != steps) else int(steps))])  # NaN -> ""

# Plots
ma_ret = moving_avg(episode_returns, 20)
ma_steps = moving_avg(steps_to_goal, 20)

plt.figure()
plt.plot(episode_returns, '.', alpha=0.4, label="Return/Episode")
plt.plot(ma_ret, linewidth=2, label="Moving Avg (20)")
plt.title("Meta-RL (RL²) – Episoden-Return über zufällige FrozenLake-Tasks")
plt.xlabel("Episode"); plt.ylabel("Return"); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "returns.png"), dpi=150, bbox_inches="tight")

plt.figure()
plt.plot(steps_to_goal, '.', alpha=0.4, label="Schritte bis Ziel")
plt.plot(ma_steps, linewidth=2, label="Moving Avg (20)")
plt.title("Meta-RL (RL²) – Schritte bis Ziel")
plt.xlabel("Episode"); plt.ylabel("Schritte (NaN = kein Ziel)"); plt.legend(); plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "steps_to_goal.png"), dpi=150, bbox_inches="tight")

if USE_TENSORBOARD:
    writer.close()

print(f"Meta-Training abgeschlossen. Output: {OUT_DIR}")

# ========================= Laden, Eval & Video =========================
def load_checkpoint(path, model, optimizer=None):
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    model.eval()
    return ckpt.get("extra", {})

def evaluate(model, n_episodes=10):
    model.eval()
    total = 0.0
    for k in range(n_episodes):
        env, _ = make_task_env(10_000 + k)
        h = torch.zeros(1, 1, HIDDEN_SIZE, device=DEVICE)
        obs, _ = env.reset()
        obs_oh = one_hot(obs, obs_size)
        last_action = torch.zeros(action_size, device=DEVICE)
        last_reward = torch.zeros(1, device=DEVICE)

        ep_ret = 0.0
        with torch.no_grad():
            for _ in range(STEPS):
                logits, value, h = net(
                    obs_oh.unsqueeze(0),
                    last_action.unsqueeze(0),
                    last_reward.view(1, 1),
                    h
                )
                action = torch.distributions.Categorical(logits=logits).sample()
                obs_next, reward, term, trunc, _ = env.step(action.item())
                ep_ret += reward
                obs_oh = one_hot(obs_next, obs_size)
                last_action = one_hot(action.item(), action_size)
                last_reward = torch.tensor([reward], dtype=torch.float32, device=DEVICE)
                if term or trunc:
                    break
        env.close()
        total += ep_ret
        print(f"Eval Episode {k+1}: Return={ep_ret}")
    print(f"Ø Return über {n_episodes} Eval-Episoden: {total/n_episodes:.3f}")

# Beispiel: bestes Modell laden und evaluieren
if os.path.exists(os.path.join(OUT_DIR, "best.pt")):
    load_checkpoint(os.path.join(OUT_DIR, "best.pt"), net)
evaluate(net, n_episodes=10)

# Video mitschneiden (eine Episode)
from gymnasium.wrappers import RecordVideo
def record_one_episode(model, video_dir):
    # Neue Task + Env mit Render für Video
    desc = generate_random_map(size=MAP_SIZE, p=MAP_P)
    env = gym.make("FrozenLake-v1", desc=desc, is_slippery=SLIPPERY, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda e: True)
    h = torch.zeros(1, 1, HIDDEN_SIZE, device=DEVICE)

    obs, _ = env.reset()
    obs_oh = one_hot(obs, obs_size)
    last_action = torch.zeros(action_size, device=DEVICE)
    last_reward = torch.zeros(1, device=DEVICE)

    with torch.no_grad():
        for _ in range(STEPS):
            logits, value, h = model(
                obs_oh.unsqueeze(0),
                last_action.unsqueeze(0),
                last_reward.view(1, 1),
                h
            )
            action = torch.distributions.Categorical(logits=logits).sample()
            obs_next, reward, term, trunc, _ = env.step(action.item())
            obs_oh = one_hot(obs_next, obs_size)
            last_action = one_hot(action.item(), action_size)
            last_reward = torch.tensor([reward], dtype=torch.float32, device=DEVICE)
            if term or trunc:
                break
    env.close()

record_one_episode(net, os.path.join(OUT_DIR, "videos"))
print(f"Video gespeichert unter: {os.path.join(OUT_DIR, 'videos')}")
