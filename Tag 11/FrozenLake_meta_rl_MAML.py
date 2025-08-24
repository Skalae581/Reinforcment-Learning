from random import random

import numpy as np
import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map
from torch import nn, optim
import torch
import torch.nn.functional as F

SEED = 0
epsilon = "auto"
np.random.seed(SEED)
torch.manual_seed(SEED)

trial_numbers = 200
horizons = 50  # episodes per trial
max_steps_per_episode = 300
inner_steps = 5


def make_env(render=False):
	kwargs = {"desc": generate_random_map(size=4)}
	if render:
		kwargs["render_mode"] = "human"
	return gym.make("FrozenLake-v1", **kwargs)


# ============================ Env / MDP ============================
probe_env = make_env(render=False)
num_states = probe_env.observation_space.n
num_actions = probe_env.action_space.n

policy_net = nn.Sequential(
	nn.Linear(num_states, 128),
	nn.ReLU(),
	nn.Linear(128, num_actions),
	nn.Softmax(dim=-1)
)

control_net = nn.Sequential(
	nn.Linear(num_states, 128),
	nn.ReLU(),
	nn.Linear(128, num_actions),
	nn.Softmax(dim=-1)
)
control_net.load_state_dict(policy_net.state_dict())  # initialize controller = policy

lr = 3e-4
optimizer = optim.Adam(policy_net.parameters(), lr=lr)
meta_lr = 1e-3
controller_optimizer = optim.Adam(control_net.parameters(), lr=meta_lr)


def to_one_hot(s: int) -> torch.Tensor:
	return F.one_hot(torch.tensor(int(s)), num_classes=num_states).float()


def policy_direct(state_vec: torch.Tensor) -> int:
	with torch.no_grad():
		probs = policy_net(state_vec)
	return torch.multinomial(probs, num_samples=1).item()


def policy_eps(state_vec: torch.Tensor, eps: float) -> int:
	if random() < eps:
		return np.random.randint(num_actions)
	return policy_direct(state_vec)


def own_reward(next_state_nr: int, raw_reward: float) -> float:
	# shaped reward for FrozenLake 4x4
	if next_state_nr == 15:  # goal state
		return 10.0
	elif next_state_nr == 0:  # start state
		return 0.0
	elif next_state_nr in [5, 7, 11, 12]:  # holes
		return -1.0
	else:
		return 0.01 + float(raw_reward)


def reinforce_loss(trajectory, gamma: float = 0.99) -> torch.Tensor:
	G = 0.0
	losses = []
	for (s_vec, a, r) in reversed(trajectory):
		G = r + gamma * G
		probs = policy_net(s_vec)  # π_θ(a|s)
		logp = torch.log(probs[a] + 1e-9)
		losses.append(-logp * torch.tensor(G))
	loss = torch.stack(losses).sum()
	return torch.clamp(loss, -10.0, 10.0)

# Store the latest evaluation trajectory for meta-loss computation
last_eval_traj = None


def run_episode(env, eps: float) -> tuple[list[tuple[torch.Tensor, int, float]], float]:
	state, _ = env.reset(seed=SEED)
	done = False
	trajectory = []
	total = 0.0
	steps = 0
	while not done and steps < max_steps_per_episode:
		s_vec = to_one_hot(state)
		action = policy_eps(s_vec, eps)
		next_state, reward, terminated, truncated, _ = env.step(action)
		shaped = own_reward(next_state, reward)
		trajectory.append((s_vec, action, shaped))
		state = next_state
		done = terminated or truncated
		total += shaped
		steps += 1
	return trajectory, total


def inner_loop(task_env, phi):
	# load controller params into fast policy
	policy_net.load_state_dict(phi)
	# adapt with a few REINFORCE episodes
	for k in range(inner_steps):
		eps = max(0.0, 1.0 - 2.0 * k / inner_steps)
		traj, _ = run_episode(task_env, eps)
		loss = reinforce_loss(traj)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	# evaluate adapted policy without exploration
	traj_eval, eval_gain = run_episode(task_env, eps=0.0)
	global last_eval_traj
	last_eval_traj = traj_eval
	return eval_gain


def update_control_net():
	"""Gradient-based meta-step (FOMAML).
	Uses the REINFORCE loss on the evaluation trajectory collected *after* inner
	adaptation to update the controller's parameters via the fast-policy grads.
	Approximation: ∇_φ J(D', π_{θ'(D)}) ≈ ∇_{θ'} J(D', π_{θ'}) with θ' frozen.
	"""
	global last_eval_traj
	if last_eval_traj is None:
		return
	# Compute meta-loss on adapted policy parameters (policy_net)
	for p in policy_net.parameters():
		p.grad = None
	loss_meta = reinforce_loss(last_eval_traj)
	loss_meta.backward()
	# Map fast-policy gradients onto controller parameters and step
	for p in control_net.parameters():
		p.grad = None
	for p_ctrl, p_fast in zip(control_net.parameters(), policy_net.parameters()):
		if p_fast.grad is not None:
			p_ctrl.grad = p_fast.grad.detach().clone()
	controller_optimizer.step()
	controller_optimizer.zero_grad()


def update_control_net_gradient_free():
	"""First‑order meta‑update: φ ← φ + β (θ' − φ)
	Using Reptile/FOMAML-style update to approximate ∇_φ E[J(D, π_θ')].
	"""
	meta_lr = 0.1
	with torch.no_grad():
		for p_ctrl, p_fast in zip(control_net.parameters(), policy_net.parameters()):
			p_ctrl.add_(meta_lr * (p_fast - p_ctrl))


def new_task(done_training: bool):
	return make_env(render=done_training)


def outer_loop():
	done_training = False
	for trial_number in range(trial_numbers):
		gains = 0
		for horizon in range(horizons):
			if trial_number == trial_numbers - 1:
				done_training = True
			env = new_task(done_training)
			phi = control_net.state_dict()
			gain = inner_loop(env, phi)  # adapt on task i
			gains += gain
			update_control_net()  # gradient-based meta‑step
		print(f"trial {trial_number + 1}: eval_gain≈{gains:.2f}")


if __name__ == "__main__":
	outer_loop()
