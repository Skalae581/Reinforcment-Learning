# Factory I/O Reinforcement Learning mit Modbus und PPO

Dieses Projekt demonstriert die Verwendung von Reinforcement Learning (PPO-Algorithmus) zur Steuerung einer simulierten Produktionsumgebung in [Factory I/O](https://factoryio.com/). Die Kommunikation erfolgt über das Modbus TCP Protokoll.

---

## 🔧 Voraussetzungen

- Python 3.8+
- [Factory I/O](https://factoryio.com/) mit aktiviertem **Modbus TCP Server**
- Modbus-Schnittstelle konfiguriert auf `127.0.0.1:502`
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [pyModbusTCP](https://github.com/sourceperl/pyModbusTCP)
- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- matplotlib (für Render-Visualisierung)

Installation (Empfohlen in virtuellem Environment):
```bash
pip install stable-baselines3[extra] pyModbusTCP gymnasium matplotlib

