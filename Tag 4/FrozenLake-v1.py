# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:55:18 2025

@author: TAKO
"""

import gymnasium as gym
import numpy as np
import random
import time

def q_learning_discrete(env_name='FrozenLake-v1', episodes=1000):
    env = gym.make(env_name, render_mode="human", is_slippery=False)
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    rewards = []

    # Definiere benutzerdefinierte Rewards
    HOLE_PENALTY = -5
    GOAL_REWARD = 100
    STEP_PENALTY = -0.1

    hole_states = [5, 7, 11, 12]  # Löcher im 4x4 FrozenLake
    goal_state = 15

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            # Epsilon-greedy Policy
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # --- Reward Shaping ---
            if next_state in hole_states:
                reward = HOLE_PENALTY
            elif next_state == goal_state:
                reward = GOAL_REWARD
            else:
                reward = STEP_PENALTY  # kleine Strafe für jeden Schritt

            # Q-Learning Update
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_delta = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_delta

            # Debug-Ausgabe
            print(f"Episode {ep+1}, Step {step}, State {state}, Action {action}, Reward {reward:.3f}, Total Reward {total_reward:.3f}")

            state = next_state
            total_reward += reward

            env.render()
            time.sleep(0.05)
            step += 1

        # Epsilon-Decay
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {ep+1}: Durchschnittsreward (letzte 100) = {avg_reward:.3f}")

    env.close()

    print("\nGelernt Q-Tabelle:")
    for state_idx in range(q_table.shape[0]):
        print(f"Zustand {state_idx}: {np.round(q_table[state_idx], 2)}")

    return q_table, rewards


if __name__ == "__main__":
    q_table, rewards = q_learning_discrete()
