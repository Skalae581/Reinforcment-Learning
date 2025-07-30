# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:43:26 2025

@author: TAKO
"""

from random import random,max

class Casino:
		def __init__(self):
				self.bandits = [2,5,9,4,7] # maximaler Gewinn pro Bandit

		def step(self, action, state=None): # wähle Automat 0…5
			return self.bandits[action] * random()

env = Casino()
# Aufgabe: Wähle den Automaten mit dem höchsten Gewinn PER POLICY FINDEN!!
# Tipps: mitzählen, wie oft welcher Automat gewählt wurde
# Tipps: epsilon-greedy policy!!

epsilon = "automatic see below"  # Exploration rate for epsilon-greedy policy

Besuche = [0,0,0,0,0] # Dictionary to count visits to each bandit
Gewinne = [0,0,0,0,0] # Dictionary to accumulate rewards for each bandit
DurchschnittsGewinn = [0, 0, 0, 0, 0] # Q-Werte für jeden Bandit  "Erwartete Belohnung pro Bandit"

def our_policy(epsilon):
		action = int(random() * len(env.bandits))  # Random action
		return action

Averages = [] # zum Plotten der Gewinne
def train(epochs=1000):
		total_reward = 0
		for step in range(epochs):
			epsilon = 1 - step / epochs  # Decrease epsilon over time
			action = our_policy(epsilon)
			reward = env.step(action)
			total_reward += reward
			Besuche[action] += 1
			Gewinne[action] += reward
			DurchschnittsGewinn[action] = Gewinne[action] / Besuche[action]
            mGewinn=max(...Gewinne)
            bester_B = Gewinne.index(mGewinn)
            
            
			# Debug:
			if step % 100 == 0:
				average_reward = total_reward / (step + 1)  # Erfolg messen mit moving average reward
				Averages.append(average_reward)
				print(f"Episode {step}: Action {action}, Average Reward {average_reward:.2f}")

train()


# plot
import matplotlib.pyplot as plt
def plot_results():
		plt.figure(figsize=(10, 5))
		plt.plot(Averages, label='Durchschnitts Gewinn')
		plt.xlabel('Episodes * 100')
		plt.ylabel('Gewinn')
		plt.title('Durchschnitts Gewinn über Episoden')
		plt.legend()
		plt.grid()
		plt.show()
plot_results()