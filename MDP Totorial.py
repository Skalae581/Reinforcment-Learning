# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:25:49 2025

@author: TAKO
"""

states = ['S', 'A', 'B', 'G']
actions = ['left', 'right']
gamma = 0.9
theta = 1e-4  # Abbruchbedingung (kleinste Wertänderung)
# Übergänge: (state, action) → [(next_state, probability)]
P = {
    ('S', 'right'): [('A', 1.0)],
    ('A', 'right'): [('B', 1.0)],
    ('B', 'right'): [('G', 1.0)],
    ('A', 'left'):  [('S', 1.0)],
    ('B', 'left'):  [('A', 1.0)]
}

# Belohnungen: (state, action, next_state) → reward
R = {
    ('S', 'right', 'A'): 0,
    ('A', 'right', 'B'): 0,
    ('B', 'right', 'G'): 10,
    ('A', 'left', 'S'): -1,
    ('B', 'left', 'A'): -1
}

# Value-Funktion (V(s))
V = {s: 0 for s in states}
# Policy π(s) → beste Aktion im Zustand
policy = {s: None for s in states if s != 'G'}
def value_iteration():
    while True:
        delta = 0
        for s in states:
            if s == 'G':
                continue  # Zielzustand überspringen

            v = V[s]
            max_q = float('-inf')
            best_a = None

            for a in actions:
                transitions = P.get((s, a), [])
                q_sa = 0
                for (s_next, prob) in transitions:
                    reward = R.get((s, a, s_next), 0)
                    q_sa += prob * (reward + gamma * V[s_next])

                if q_sa > max_q:
                    max_q = q_sa
                    best_a = a

            V[s] = max_q
            policy[s] = best_a
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break
value_iteration()

print("🔢 Wertfunktion:")
for s in V:
    print(f"V({s}) = {V[s]:.2f}")

print("\n🎯 Optimale Policy:")
for s in policy:
    print(f"π({s}) = {policy[s]}")
