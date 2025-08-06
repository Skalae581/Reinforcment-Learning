import gymnasium as gym
import panda_gym

# PandaReach-Environment erstellen
env = gym.make("PandaReach-v3", render_mode="human")

# Reset der Umgebung
obs, info = env.reset()

done = False
while not done:
    # Zufällige Aktionen zum Testen
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
