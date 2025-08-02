from pyModbusTCP.client import ModbusClient
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FactoryIOEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Modbus TCP Client initialisieren
        self.client = ModbusClient(host="127.0.0.1", port=502, auto_open=True)

        # Beobachtungsraum: 10 Diskrete Inputs (0 oder 1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        # Aktionsraum: 4 mögliche Aktionen (Coil 0-3 setzen)
        self.action_space = spaces.Discrete(4)

        self.state = np.zeros(10, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # State initialisieren
        self.state = np.zeros(10, dtype=np.float32)

        try:
            inputs = self.client.read_discrete_inputs(0, 10)
            if inputs is not None:
                self.state = np.array(inputs, dtype=np.float32)
        except Exception as e:
            print("Modbus Fehler im reset:", e)
            # State bleibt 0

        return self.state, {}

    def step(self, action):
        # Reset aller Coils auf False
        for coil_addr in range(self.action_space.n):
            try:
                self.client.write_single_coil(coil_addr, False)
            except Exception as e:
                print(f"Modbus Fehler beim Zurücksetzen Coil {coil_addr}:", e)

        # Aktion ausführen, falls gültig
        if 0 <= action < self.action_space.n:
            try:
                self.client.write_single_coil(action, True)
            except Exception as e:
                print(f"Modbus Fehler beim Setzen Coil {action}:", e)
        else:
            print("Ungültige Aktion:", action)

        # Beobachtung aktualisieren
        try:
            inputs = self.client.read_discrete_inputs(0, 10)
            if inputs is not None:
                self.state = np.array(inputs, dtype=np.float32)
            else:
                self.state = np.zeros(10, dtype=np.float32)
        except Exception as e:
            print("Modbus Fehler im step:", e)
            self.state = np.zeros(10, dtype=np.float32)

        # Beispiel Reward: Belohne, wenn Input 0 gesetzt ist
        reward = 1.0 if self.state[0] == 1 else 0.0

        # Beispiel: keine Terminierung (immer False), kann je nach Task angepasst werden
        terminated = False
        truncated = False

        info = {}

        return self.state, reward, terminated, truncated, info



    def render(self, mode="human"):
        import matplotlib.pyplot as plt
        plt.clf()  # Clear previous plot
        plt.bar(range(len(self.state)), self.state, color='green')
        plt.ylim(0, 1)
        plt.title("Factory Inputs")
        plt.xlabel("Input Index")
        plt.ylabel("Status")
        plt.pause(0.001)
    def close(self):
        # Modbus Verbindung schließen, wenn nötig
        if self.client.is_open():
            self.client.close()
        print("Umgebung geschlossen")


if __name__ == "__main__":
    env = FactoryIOEnv()

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)

    env.close()
