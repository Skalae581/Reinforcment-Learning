# import gym
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

nr_training_steps=2200
gamma=0.99
lr=3e-4

# Gym Environment
# env = gym.make('CartPole-v1', render_mode="human")  # pole with visualization
env = gym.make('CartPole-v1')  # pole without visualization

# Policy Network
observation_size=4 # pole state: (position, velocity, angle, angular-velocity)
hidden_size = 128
n_actions = 2 # left / right
policy = nn.Sequential( # batch wird bei pytorch IMPLIZIT mit durchgeschliffen!
						nn.Linear(observation_size, hidden_size), # keine batch size mitgeben
						nn.ReLU(),
						nn.Linear(hidden_size, n_actions),
						nn.Softmax(dim=-1)
				)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# Learned Dynamics Model!
model = nn.Sequential(
		nn.Linear(observation_size + n_actions, hidden_size),
		nn.ReLU(),
		nn.Linear(hidden_size, observation_size) # ohne reward, ansonsten +1
)
model_optimizer = optim.Adam(model.parameters(), lr=lr)

def train_model(action, state, state_tensor):
	# Train dynamics model with observed transition
	action_tensor = torch.nn.functional.one_hot(torch.tensor(action), num_classes=n_actions).float()
	state_and_action = torch.cat([state_tensor, action_tensor]) # concatenate state + action
	predicted_next_state = model(state_and_action)  # predict next state
	target_next_state = torch.tensor(state, dtype=torch.float32)
	# loss = ( predicted - target ) ^2
	model_loss = nn.functional.mse_loss(predicted_next_state, target_next_state) # MSE
	model_optimizer.zero_grad()
	model_loss.backward()
	model_optimizer.step()

def simulate():
	# Simulate trajectory(episode) with model
	model_state = torch.tensor(env.reset()[0], dtype=torch.float32)
	model_rewards = []
	model_probs = []
	max_steps = 500
	for _ in range(max_steps):  # simulate same length
		try:
			model_prob = policy(model_state) # action probabilities from model state
			model_action = torch.multinomial(model_prob, num_samples=1).item()
			model_action_tensor = torch.nn.functional.one_hot(torch.tensor(model_action), num_classes=n_actions).float()
			state_and_action = torch.cat([model_state, model_action_tensor]) # concatenate state + action
			model_state = model(state_and_action).detach() # << statt env.step(action) in real environment
			angle = model_state[2].item()  # angle is the third element in the state
			if angle < -0.21 or angle > 0.21:
				break
			# hier ^^^ OHNE reward, wir wissen er ist immer 1.0
			model_reward = 1.0  # constant reward for CartPole until done übertrieben dies mitzulernen
			log_prob = torch.log(model_prob[model_action])
			# log_prob = torch.clamp(log_prob, -1, 1.5)  # clamp log probabilities for numerical stability
			model_probs.append(log_prob)
			model_rewards.append(model_reward)
		except Exception as e:
			print("Simulation error:", e)
			# break
	return model_probs,model_rewards

def update_policy(log_probs, rewards):
	# Update policy with REINFORCE algorithm, jeder andere algo tut es auch!
	# Compute returns
	returns = []
	G = 0  # Gain over whole episode
	for r in reversed(rewards):
		G = r + gamma * G  # discounted early rewards
		returns.insert(0, G)
	returns = torch.tensor(returns)

	# Normalize returns for numerical stability
	returns = (returns - returns.mean()) / (returns.std() + 1e-8)
	# log_probs = torch.clamp(log_probs, -1, 1.5)
	# Policy gradient update
	loss = -log_probs @ returns  # skalar product (log1*G1 + log2*G2 + ...)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss.item()

run_lengths=[]
# Training REINFORCE
for training_step in range(nr_training_steps):
		state = env.reset()[0]
		log_probs = [] # action probabilities (log'ed)
		rewards = [] # rewards for each action taken

		done = False
		steps = 0
		# play_game(transition=env.step(action))
		while not done: # ganz normale episode
				steps = steps + 1
				state_tensor = torch.tensor(state, dtype=torch.float32)
				probs = policy(state_tensor)
				action = torch.multinomial(probs, num_samples=1).item()
				log_probs.append(torch.log(probs[action]))
				state, reward, terminated, truncated, info = env.step(action) # action taken!
				rewards.append(reward)
				done = terminated or truncated # pole is down or time is up!
				train_model(action, state, state_tensor)
		# per replay
		# train_model(action, state, state_tensor)
		loss = update_policy(torch.stack(log_probs), rewards) # classic

		if training_step > 1000:  # after some steps, use model to simulate
			model_probs, model_rewards = simulate()  # Simulate with model!
			loss += update_policy(torch.stack(model_probs), model_rewards)

		run_lengths.append(steps)


		# live plot of moving average run lengths
		if training_step % 100 == 0:
				print("step", training_step, "loss", loss, "steps", steps)
				print("mean run length", sum(run_lengths)/len(run_lengths))
				if len(run_lengths) >= 40:
						ma = torch.tensor(run_lengths).float().unfold(0, 40, 1).mean(dim=1)
						# plot()

def plot():
	plt.plot(run_lengths)
	plt.savefig("run_lengths.png")
	plt.show()
	plt.pause(0.001)

plot()