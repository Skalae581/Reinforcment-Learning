import gymnasium as gym

# Wichtig: zuerst dein Paket importieren, damit register() ausgeführt wird
import pickomino_env  # falls du ein eigenes Env registrierst

env_ids = sorted(gym.envs.registry.keys())
print(env_ids)

