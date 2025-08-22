# Gym-Registrierung & SB3-PPO: Von der Env zum Training (mit TensorBoard)

## 1) Ziel & Kontext
- Eigene Gymnasium-Umgebung („Pickomino“) registrieren, korrekt wrappen und mit Stable-Baselines3 (PPO) trainieren.
- Häufige Fehler vermeiden (Nested-Obs, Tuple-Actions, falsche Obs-Werte).
- Training via TensorBoard sichtbar machen.

---

## 2) Agenda
1. Warum Registrierung?
2. Projektstruktur
3. Registrierungscode
4. Observation/Action Spaces – Do’s & Don’ts
5. Wrapper-Pipeline (robuste Variante)
6. PPO-Setup & Training
7. TensorBoard starten
8. Troubleshooting (konkrete Fehler)
9. Nächste Schritte
10. **Rollout-Kurven – Auswertung** ✅

---

## 3) Warum Registrierung?
- `gym.make("Pickomino-v0")` funktioniert nur nach vorheriger Registrierung.
- Registrierung macht die Env global auffindbar und legt `id`, `entry_point`, `max_episode_steps` etc. fest.

---

## 4) Empfohlene Projektstruktur
```
pickomino/
├─ pickomino/
│  ├─ __init__.py          # register(...) oder nur Exporte
│  ├─ envs/
│  │  ├─ __init__.py       # from .pickomino_env import PickominoEnv
│  │  └─ pickomino_env.py  # class PickominoEnv(gym.Env)
├─ setup.py / pyproject.toml (optional, wenn als Paket installiert)
└─ train_ppo.py            # Training (SB3, Wrapper, TensorBoard)
```

---

## 5) Registrierungscode (Minimal)
**Variante A: Ad-hoc** (vor `gym.make`)
```python
from gymnasium.envs.registration import register
register(
    id="Pickomino-v0",
    entry_point="pickomino.envs:PickominoEnv",
    max_episode_steps=200,
)
```
**Variante B: In `pickomino/__init__.py`** (beim Import automatisch)
```python
from gymnasium.envs.registration import register
register(id="Pickomino-v0", entry_point="pickomino.envs:PickominoEnv")
# dann im Training: import pickomino; gym.make("Pickomino-v0")
```

---

## 6) Observation & Action Spaces – Do’s & Don’ts
- **Obs:** Box (Vektor) oder 1‑Level‑Dict. Keine Dict-in-Dict/Tuple-in-Dict (sonst vorher flatten).
- **Actions:** SB3 unterstützt `Discrete`, `MultiDiscrete`, `MultiBinary`, `Box`. Tuple-Actions vorher konvertieren.
- **Step/Reset geben Daten zurück** (np.ndarray/int/dict), niemals Spaces.

---

## 7) Wrapper-Pipeline (robust)
- `TupleToMultiDiscrete` → `Tuple(Discrete,...)` → `MultiDiscrete` für SB3; Rückkonvertierung beim `step`.
- `sanitize_obs(...)` + `SafeFlattenToBox` → repariert Typ-/Range-/Shape-Fehler und flatten zu 1D‑Box.
- `Monitor` → Episoden-Return und -Länge für Logger/TensorBoard.

---

## 8) PPO-Setup & Training (Ausschnitt)
```python
env = make_env(0)
from stable_baselines3.common.env_checker import check_env
check_env(env, warn=True)

from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, tensorboard_log="runs", verbose=1)
model.learn(10_000, tb_log_name="PickominoPPO")
```

---

## 9) TensorBoard starten (Windows CMD)
```cmd
cd /d "C:\Users\kosch\Desktop\Reinforcment Learning\pickomino"
tensorboard --logdir runs --reload_interval 3 --port 6006
```
Im Browser öffnen: `http://localhost:6006/`.

---

## 10) Troubleshooting (kurz)
- **„Nested observation spaces …“** → 1‑Level‑Dict nutzen oder mit `SafeFlattenToBox` flatten.
- **`TypeError … not 'Box'` beim Flatten** → Env gibt Space als Wert zurück → `sanitize_obs` fängt es ab; langfristig Env fixen.
- **`IndexError … Discrete`** → Out-of-Range‑Werte → `sanitize_obs` mappt in Range; langfristig korrekt erzeugen.

---


## 11) Nächste Schritte
- Falls Bilder: 1‑Level‑Dict (`{"image": Box(H,W,C), "state": Box(n,)}`) + `MultiInputPolicy` statt Flatten.
- `EvalCallback` einsetzen (rauschärmere Metriken), Lernrate per Schedule feintunen.
- Optional: `make_vec_env(..., n_envs>1)` für Parallelisierung.

---

## 12) Rollout‑Kurven – Auswertung (deine TB‑Screenshots)
**ep_len_mean (links)**  
- Start ~1.5, ab ~20k steiler Anstieg, Plateau ~3.7–4.0 → Episoden werden im Mittel deutlich länger.

**ep_rew_mean (rechts)**  
- Bis ~25k nahe 0, dann Lernschub auf ~1.2–1.3 und Stabilisierung → Politik findet signifikant bessere Strategie.

**Interpretation**  
- Typischer Breakpoint: anfängliche Exploration, dann Policy‑Verbesserung und Plateau.
- Da Länge **und** Reward steigen, ist „länger leben“ hier **gut** (mehr Punkte/erfolgreichere Trajektorien).

**Empfehlungen**  
- **Feintuning mit LR‑Decay:**
```python
model = PPO(..., learning_rate=lambda frac: 3e-4 * frac, ...)
```
- **Rauscharme Evaluation & Best‑Model‑Save:**
```python
from stable_baselines3.common.callbacks import EvalCallback
eval_cb = EvalCallback(make_env(123), eval_freq=5000, n_eval_episodes=20,
                      deterministic=True, best_model_save_path="models", log_path="eval")
model.learn(200_000, tb_log_name="PickominoPPO", callback=eval_cb)
```
- **Plateau schieben (optional):** `ent_coef` ↓ (z. B. 0.01→0.005), `n_steps` ↑ (2048→4096), `clip_range` feinjustieren (0.15–0.25).
- **TimeLimit prüfen:** Häufung am Max‑Step ⇒ `truncated`‑Ende; ggf. `max_episode_steps` anpassen oder Reward‑Shaping.

---

## 13) Komplettes Trainings‑Snippet (kompakt)
```python
env = make_env(0)
check_env(env, warn=True)
model = PPO("MlpPolicy", env, tensorboard_log="runs", verbose=1)
model.learn(10_000, tb_log_name="PickominoPPO")  # optional: callback=TBCallback()
```

> **Screenshots** der TensorBoard‑Plots (ep_len_mean / ep_rew_mean) hier einfügen.
