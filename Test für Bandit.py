# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 11:22:06 2025

@author: TAKO
"""

import torch
import random

# ==== MDP-Definition ====
states = ['S', '1', 'G', 'T']  # S = Start, G = Goal, T = Trap
actions = ['right', 'down', 'left']

# Übergangswahrscheinlichkeiten: (state, action) -> [(next_state, prob)]
transitions = {
    ('S', 'right'): [('1', 1.0)],
    ('1', 'right'): [('G', 1.0)],
    ('1', 'down'): [('T', 1.0)],
    ('1', 'left'): [('S', 1.0)],
}

# Belohnungsfunktion
rewards = {
    'G': 10.0,
    'T': -10.0,
    'S': 0.0,
    '1': 0.0
}

# Diskontfaktor
gamma = 0.9

# ==== Value Iteration ====
V = {s: torch.tensor(0.0) for s in states}  # Startwerte
policy = {s: None for s in states if s != 'G' and s != 'T'}  # Optimale Aktion

def one_step_lookahead(state, V):
    A = {}
    for a in actions:
        outcomes = transitions.get((state, a), [])
        total = 0
        for (next_state, prob) in outcomes:
            r = rewards[next_state]
            total += prob * (r + gamma * V[next_state])
        A[a] = total
    return A

# Value Iteration
for i in range(10):
    for s in ['S', '1']:  # Terminale Zustände ignorieren
        action_values = one_step_lookahead(s, V)
        best_action = max(action_values, key=action_values.get)
        V[s] = torch.tensor(action_values[best_action])
        policy[s] = best_action

# ==== Ergebnisse ====
print("🔢 Wertfunktion:")
for s in V:
    print(f"  V({s}) = {V[s]:.2f}")

print("\n🎯 Optimale Policy:")
for s in policy:
    print(f"  π({s}) = {policy[s]}")
