import numpy as np
import matplotlib.pyplot as plt
import random


GRID_SIZE = 4

TERMINAL_STATE = (0, 0)
DISCOUNT_FACTOR = .99 # bei 1 läuft der Algorithmus im Kreis: nicht konvergierend
ACTIONS = ['up', 'down', 'left', 'right']
ACTION_EFFECTS = {
		'up': (-1, 0), # (y, x) -> (y-1, x) position change
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
rewards[3, 3] = -10


# Parameters
ALPHA = 0.1
EPSILON = 0.1
EPISODES = 5000

# Q-table initialization
Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

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

def epsilon_greedy(state):
	if random.random() < EPSILON:
		return random.choice(range(len(ACTIONS)))
	else:
		y, x = state
		return np.argmax(Q[y, x])

def update_q_values(state, action_idx, reward, next_state):
	y, x = state
	ny, nx = next_state
	best_next = np.max(Q[ny, nx])
	Q[y, x, action_idx] += ALPHA * (reward + DISCOUNT_FACTOR * best_next - Q[y, x, action_idx])


# Q-learning loop
for episode in range(EPISODES):
	state = (GRID_SIZE - 1, GRID_SIZE - 1)
	while not is_terminal(state):
		action_idx = epsilon_greedy(state)
		action = ACTIONS[action_idx]
		next_state, reward = step(state, action)
		update_q_values(state, action_idx, reward, next_state)
		state = next_state

# Extract final policy
policy = np.full((GRID_SIZE, GRID_SIZE), '', dtype=object)
for y in range(GRID_SIZE):
	for x in range(GRID_SIZE):
		if is_terminal((y, x)):
			policy[y, x] = '🟩'
		else:
			best_action = np.argmax(Q[y, x])
			policy[y, x] = ACTION_SYMBOLS[ACTIONS[best_action]]
for y in range(GRID_SIZE):
    for x in range(GRID_SIZE):
        print(f"State ({y},{x}):")
        for i, action in enumerate(ACTIONS):
            print(f"  {ACTION_SYMBOLS[action]} = {Q[y,x,i]:.2f}")

# --- SARSA Algorithmus ---
Q_sarsa = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

for episode in range(EPISODES):
    state = (GRID_SIZE - 1, GRID_SIZE - 1)  # Start unten rechts
    action_idx = epsilon_greedy(state)

    while not is_terminal(state):
        action = ACTIONS[action_idx]
        next_state, reward = step(state, action)

        # Nächste Aktion (On-Policy)
        next_action_idx = epsilon_greedy(next_state)

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
for i, action in enumerate(ACTIONS):
    plt.figure()
    plt.title(f"Q-Werte für Aktion: {action}")
    plt.imshow(Q[:,:,i], cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Display policy
for row in policy:
	print(' '.join(row))
    