# -*- coding: utf-8 -*-
"""
Refactored Multi-Armed Bandit mit zwei Policies
@author: TAKO
"""

from random import random, randint
import matplotlib.pyplot as plt
import os

class Casino:
    def __init__(self):
        self.bandits = [2, 5, 9, 4, 7]  # Maximaler Gewinn pro Bandit

    def step(self, action, state=None):
        return self.bandits[action] * random()

# === Globale Initialisierung ===
env = Casino()
anzahl_banditen = len(env.bandits)

# Statistik
Besuche = [0] * anzahl_banditen
Gewinne = [0.0] * anzahl_banditen
DurchschnittsGewinn = [0.0] * anzahl_banditen
Averages = []

# === Policies ===
def random_policy(epsilon):
    return randint(0, anzahl_banditen - 1)

def epsilon_greedy_policy(epsilon):
    if random() < epsilon:
        return randint(0, anzahl_banditen - 1)
    else:
        return DurchschnittsGewinn.index(max(DurchschnittsGewinn))

# === Training ===
def train(policy_fn, epochs=1000):
    global Besuche, Gewinne, DurchschnittsGewinn, Averages
    
    Besuche = [0] * anzahl_banditen
    Gewinne = [0.0] * anzahl_banditen
    DurchschnittsGewinn = [0.0] * anzahl_banditen
    Averages = []

    total_reward = 0
    for step in range(epochs):
        epsilon = max(0.01, 1 - step / epochs)
        action = policy_fn(epsilon)
        reward = env.step(action)

        total_reward += reward
        Besuche[action] += 1
        Gewinne[action] += reward
        DurchschnittsGewinn[action] = Gewinne[action] / Besuche[action]

        if step % 100 == 0:
            avg = total_reward / (step + 1)
            Averages.append(avg)
            bester = Gewinne.index(max(Gewinne))
            print(f"Episode {step}: Action {action}, AvgReward {avg:.2f}, Bester: {bester}, Gewinn: {Gewinne[bester]:.2f}")

# === Plot-Funktionen ===
def plot_gesamtgewinne():
    plt.figure(figsize=(10, 5))
    plt.bar([f"Bandit {i}" for i in range(anzahl_banditen)], Gewinne, color='skyblue')
    plt.title("Gesamtgewinn pro Bandit")
    plt.ylabel("Gewinn")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("gesamtgewinn.png")
    plt.show()

def plot_besuche():
    plt.figure(figsize=(10, 5))
    plt.bar([f"Bandit {i}" for i in range(anzahl_banditen)], Besuche, color='lightgreen')
    plt.title("Zuganzahl pro Bandit")
    plt.ylabel("Züge")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig("besuche.png")
    plt.show()

def plot_q_werte():
    plt.figure(figsize=(10, 5))
    plt.plot([f"Bandit {i}" for i in range(anzahl_banditen)], DurchschnittsGewinn, marker='o', label='Q-Wert')
    plt.title("Durchschnittlicher Gewinn (Q-Wert)")
    plt.ylabel("Ø Gewinn")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("q_werte.png")
    plt.show()

def plot_best_bandit():
    bester = Gewinne.index(max(Gewinne))
    plt.figure(figsize=(10, 5))
    plt.plot([f"Bandit {i}" for i in range(anzahl_banditen)], Gewinne, marker='o', label='Gesamtgewinn')
    plt.plot(f"Bandit {bester}", Gewinne[bester], 'ro', label=f'Max: Bandit {bester} ({Gewinne[bester]:.2f})')
    plt.title("Gesamtgewinn mit markiertem Top-Bandit")
    plt.ylabel("Gewinn")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("top_bandit.png")
    plt.show()

# === Trainingslauf mit beiden Policies ===
print("--- Training mit random_policy ---")
train(random_policy, epochs=1000)
plot_gesamtgewinne()
plot_besuche()
plot_q_werte()
plot_best_bandit()

print("\n--- Training mit epsilon_greedy_policy ---")
train(epsilon_greedy_policy, epochs=1000)
plot_gesamtgewinne()
plot_besuche()
plot_q_werte()
plot_best_bandit()

# === GitHub-Vorbereitung ===
# Alle generierten Plots werden als .png gespeichert
# Dateien können nun mit Git zu GitHub gepusht werden:
# git init
# git add .
# git commit -m "Add bandit experiment with two policies"
# git remote add origin <repo-URL>
# git push -u origin main
