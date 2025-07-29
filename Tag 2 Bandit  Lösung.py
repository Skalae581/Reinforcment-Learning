

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:43:26 2025

@author: TAKO
"""

from random import random

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


Averages = []  # Zum Plotten der Gewinne

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

        # Maximalgewinn und bester Bandit berechnen
        mGewinn = max(Gewinne)
        bester_B = Gewinne.index(mGewinn)

        # Debug:
        if step % 100 == 0:
            average_reward = total_reward / (step + 1)  # Erfolg messen mit moving average reward
            Averages.append(average_reward)
            print(f"Episode {step}: Action {action}, Average Reward {average_reward:.2f}, "
                  f"Bester Bandit: {bester_B}, Gesamtgewinn: {mGewinn:.2f}")

train()

for i in range(len(Gewinne)):
    print(f"Bandit {i}: Gesamtgewinn = {Gewinne[i]:.2f} aus {Besuche[i]} Zügen")

# plot
import matplotlib.pyplot as plt

def plot_gesamtgewinne():
    banditen = [f"Bandit {i}" for i in range(len(Gewinne))]
    plt.figure(figsize=(10, 5))
    plt.bar(banditen, Gewinne, color='skyblue')
    plt.xlabel("Bandit")
    plt.ylabel("Gesamtgewinn")
    plt.title("Gesamtgewinn pro Bandit")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
plot_gesamtgewinne()

def plot_besuche():
    banditen = [f"Bandit {i}" for i in range(len(Besuche))]
    plt.figure(figsize=(10, 5))
    plt.bar(banditen, Besuche, color='lightgreen')
    plt.xlabel("Bandit")
    plt.ylabel("Anzahl der Züge")
    plt.title("Zuganzahl pro Bandit")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
plot_besuche()

def plot_bandit_q_werte():
    banditen = [f"Bandit {i}" for i in range(len(DurchschnittsGewinn))]
    
    plt.figure(figsize=(12, 6))
    plt.plot(banditen, DurchschnittsGewinn, marker='o', linestyle='-', label='Ø Gewinn (Q-Wert)')
    
    plt.xlabel("Bandit")
    plt.ylabel("Durchschnittlicher Gewinn")
    plt.title("Durchschnittlicher Gewinn pro Bandit (Q-Wert)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_bandit_q_werte()


def plot_bandit_gesamtgewinne():
    banditen = [f"Bandit {i}" for i in range(len(Gewinne))]
    max_gewinn = max(Gewinne)
    bester_bandit = Gewinne.index(max_gewinn)

    plt.figure(figsize=(12, 6))
    plt.plot(banditen, Gewinne, marker='o', linestyle='-', label='Gesamtgewinn')

    # Markiere den besten Bandit
    plt.plot(banditen[bester_bandit], max_gewinn, 'ro', label=f'Max: Bandit {bester_bandit} ({max_gewinn:.2f})')

    plt.xlabel("Bandit")
    plt.ylabel("Gesamtgewinn")
    plt.title("Gesamtgewinn pro Bandit")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
plot_bandit_gesamtgewinne()