
# 🧠 Reinforcement Learning Woche 1 – Einführung

---

## 📘 Tag 1: Grundkonzepte & Motivation

### Was ist Reinforcement Learning?

- Ein Lernverfahren, bei dem ein **Agent** durch Interaktion mit einer **Umgebung** lernt, **Handlungen** auszuführen.
- Ziel: **Maximierung kumulierter Belohnung (Reward)**.

**Begriffe:**
- **Agent:** Entscheidet und handelt.
- **Umgebung (Environment):** Gibt Zustände und Belohnungen zurück.
- **Zustand (State):** Momentane Situation.
- **Aktion (Action):** Entscheidung des Agenten.
- **Reward:** Rückmeldung vom System.
- **Policy (π):** Strategie, welche Aktion in welchem Zustand.

---

## 📗 Tag 2: Markov Decision Processes (MDP) & Value Functions

Ein **MDP** besteht aus:
- Zuständen \(S\)
- Aktionen \(A\)
- Übergangswahrscheinlichkeiten \(P(s'|s,a)\)
- Belohnungsfunktion \(R(s,a)\)
- Discount-Faktor \( \gamma \in [0,1] \)

### Value Functions
- **State-Value:**  
  \( V^{\pi}(s) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s \right] \)
- **Action-Value:**  
  \( Q^{\pi}(s,a) = \mathbb{E}_{\pi} \left[ \sum_{t=0}^{\infty} \gamma^t R_{t+1} \mid S_0 = s, A_0 = a \right] \)

---

## 📘 Tag 3: Bellman-Gleichung & Dynamic Programming

### Bellman-Gleichung:
Für eine gegebene Policy \( \pi \):
\[
V^{\pi}(s) = R(s, \pi(s)) + \gamma \sum_{s'} P(s' \mid s, \pi(s)) V^{\pi}(s')
\]

Für die optimale Policy:
\[
V^*(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s' \mid s,a)V^*(s') \right]
\]

### Policy Iteration:
1. **Policy Evaluation**
2. **Policy Improvement**
3. Wiederholen bis Konvergenz

---

## 📙 Tag 4: Exploration, Q-Learning, Off-/On-Policy

### Q-Learning (Off-Policy):
\[
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
\]

### Exploration vs. Exploitation
- **Exploration:** Neue Aktionen testen
- **Exploitation:** Die beste bekannte Aktion wählen
- **ε-greedy:** 
  - mit Wahrscheinlichkeit ε: zufällige Aktion
  - sonst: beste bekannte Aktion

### On-Policy vs. Off-Policy
- **On-Policy:** Lernen mit aktueller Policy (z. B. SARSA)
- **Off-Policy:** Lernen mit anderer Ziel-Policy (z. B. Q-Learning)

---

## 📗 Tag 5: Anwendungen & Konvergenz

### Anwendungsfelder:
- Spiele (AlphaGo, Atari)
- Robotik
- Empfehlungssysteme
- Smart Grids

### Konvergenzbedingungen:
- \( \gamma < 1 \)
- Lernrate \( \alpha \to 0 \) mit der Zeit
- Alle Zustände oft genug besucht

---

## 🛠️ Bonus: Bootstrapping

Beim Bootstrapping:
- Wertschätzung wird durch Schätzung zukünftiger Werte **verbessert**
- Typisch für **Temporal Difference (TD)** Methoden
- Z. B. in Q-Learning oder SARSA

---

## ✅ Übungsvorschlag

Implementiere ein 4x4-GridWorld mit Policy Iteration. Erstelle eine Heatmap der Wertefunktion und zeichne die aktuelle Policy ein.

=======
# Tag 3 Experimente

