# irl_frozenlake_maxmargin.py
# Apprenticeship Learning via Feature Matching / Max-Margin IRL (Abbeel & Ng)
# auf Gymnasium FrozenLake 4x4 (is_slippery=False)

# === SAFETY PRELUDE (wichtig, v.a. im Notebook) ===
# Falls im selben Kernel vorher eine PettingZoo-/Supersuit-Env in `env` steckt:

# ================================================

import numpy as np
import gymnasium as gym
from gymnasium.core import Env          # <— für den Typ-Check
import matplotlib.pyplot as plt

#SEED = 0
#np.random.seed(SEED)

# ------------------------- Gym/MDP Setup -------------------------
def make_env():
      
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False)
    #print("fresh type(env) =", type(env))
    #assert isinstance(env, Env)
    
    # jetzt erst wrappen:
    #env = RecordEpisodeStatistics(env)
   # print("wrapped type(env) =", type(env))
    #assert isinstance(env, Env)
    env.reset(seed=0)
    nS = env.observation_space.n
    nA = env.action_space.n
    P  = env.unwrapped.P  # dict[state][action] -> List[(prob, s', r, done)]
    return env, nS, nA, P

def grid_chars(env):
    desc = env.unwrapped.desc.astype(str)
    chars = []
    for r in range(desc.shape[0]):
        for c in range(desc.shape[1]):
            chars.append(desc[r, c])
    return chars

# ------------------------- RL Solver (Value Iteration) -------------------------
def value_iteration(P, nS, nA, R_s, gamma=0.99, tol=1e-9, max_iter=10000):
    """Solve MDP for deterministic optimal policy under state-reward R_s"""
    V = np.zeros(nS, dtype=float)
    for _ in range(max_iter):
        V_new = np.empty_like(V)
        for s in range(nS):
            q = np.empty(nA, dtype=float)
            for a in range(nA):
                qa = 0.0
                for (p, s2, _r_env, done) in P[s][a]:
                    qa += p * (R_s[s] + (0.0 if done else gamma * V[s2]))
                q[a] = qa
            V_new[s] = np.max(q)
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new
            break
        V = V_new
    # greedy policy
    pi = np.zeros(nS, dtype=int)
    for s in range(nS):
        q = np.empty(nA, dtype=float)
        for a in range(nA):
            qa = 0.0
            for (p, s2, _r_env, done) in P[s][a]:
                qa += p * (R_s[s] + (0.0 if done else gamma * V[s2]))
            q[a] = qa
        pi[s] = int(np.argmax(q))
    return V, pi

# ------------------------- Occupancy / Feature Expectations -------------------------
def policy_transition_matrix(P, nS, nA, Pi):
    """
    Build P_pi (nS x nS) for (possibly stochastic) policy Pi[s,a] (row-stochastic).
    If Pi is a 1D array of ints, it's deterministic.
    """
    P_pi = np.zeros((nS, nS), dtype=float)
    if Pi.ndim == 1:
        a_of_s = Pi
        for s in range(nS):
            a = a_of_s[s]
            for (p, s2, _r, _done) in P[s][a]:
                P_pi[s, s2] += p
    else:
        for s in range(nS):
            for a in range(nA):
                pa = Pi[s, a]
                if pa == 0.0:
                    continue
                for (p, s2, _r, _done) in P[s][a]:
                    P_pi[s, s2] += pa * p
    return P_pi

def discounted_state_visitation(P, nS, nA, Pi, gamma=0.99, start_state=0):
    """
    Solve d = d0 + gamma * P_pi^T d  ->  (I - gamma P_pi^T) d = d0
    """
    P_pi = policy_transition_matrix(P, nS, nA, Pi)
    I = np.eye(nS)
    d0 = np.zeros(nS, dtype=float)
    d0[start_state] = 1.0
    d = np.linalg.solve(I - gamma * P_pi.T, d0)
    return d  # (nS,)

def feature_expectations_from_policy(P, nS, nA, Pi, Phi, gamma=0.99, start_state=0):
    d = discounted_state_visitation(P, nS, nA, Pi, gamma=gamma, start_state=start_state)
    mu = Phi.T @ d
    return mu  # (feat_dim,)

def feature_expectations_from_demos(trajs, Phi, gamma=0.99):
    """
    Empirical discounted feature expectations from demonstrations (state-only features).
    """
    feat_dim = Phi.shape[1]
    mu = np.zeros(feat_dim, dtype=float)
    for traj in trajs:
        for t, (s, a, s2) in enumerate(traj):
            mu += (gamma ** t) * Phi[s]
    mu /= len(trajs)
    print(trajs)
    #print("Dim:"feat_dim)
    return mu

# ------------------------- Simplex Projection -------------------------
def project_to_simplex(v):
    """
    Euclidean projection onto the probability simplex:
    min ||x - v|| s.t. x>=0, sum x = 1
    """
    if v.ndim != 1:
        raise ValueError("v must be 1-D")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0]
    rho = rho[-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    return w

# ------------------------- Max-Margin Apprenticeship Learning -------------------------
def apprenticeship_learning_maxmargin(P, nS, nA, Phi, mu_E, gamma=0.99,
                                     start_state=0, max_iters=500, tol=1e-3, verbose=True):
    """
    Abbeel & Ng (2004) Projection Algorithm
    """
    feat_dim = Phi.shape[1]
    mu_list = []
    pi_list = []
    w = np.random.randn(feat_dim)
    w /= (np.linalg.norm(w) + 1e-8)

    for it in range(max_iters):
        # 1) Solve MDP for current weights
        R_w = Phi @ w
        _, pi_w = value_iteration(P, nS, nA, R_w, gamma=gamma)
        mu_w = feature_expectations_from_policy(P, nS, nA, pi_w, Phi, gamma=gamma, start_state=start_state)

        mu_list.append(mu_w)
        pi_list.append(pi_w)

        # 2) Best convex combo of collected mus to approx mu_E
        A = np.stack(mu_list, axis=1)  # feat_dim x k
        lam_ls, *_ = np.linalg.lstsq(A, mu_E, rcond=None)
        lam = project_to_simplex(lam_ls)
        mu_proj = A @ lam

        # 3) Direction & margin
        diff = mu_E - mu_proj
        gap = np.linalg.norm(diff)
        if verbose:
            print(f"[MM-IRL] iter={it:02d}  policies={len(mu_list)}  ||muE - mu_proj||={gap:.6f}")

        if gap < tol:
            if verbose:
                print("[MM-IRL] Converged.")
            break

        # 4) Update w (max-margin direction)
        w = diff / (np.linalg.norm(diff) + 1e-8)

    return {
        "policies": pi_list,
        "mus": mu_list,
        "lambdas": lam,
        "mu_proj": mu_proj,
        "gap": gap,
        "w": w
    }

# ------------------------- Demos & Evaluation -------------------------
def rollout_policy(env, pi, max_steps=1000):
    traj = []
    obs, _ = env.reset()
    for t in range(max_steps):
        a = int(pi[obs])
        obs2, rew, term, trunc, _ = env.step(a)
        traj.append((obs, a, obs2))
        obs = obs2
        if term or trunc:
            break
    return traj

def eval_policy(env, pi, episodes=300, max_steps=100):
    success = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            a = int(pi[obs])
            obs, rew, term, trunc, _ = env.step(a)
            if term or trunc:
                if rew > 0.0:
                    success += 1
                break
    return success / episodes

def plot_grid(values, title):
    grid = values.reshape(4,4)
    plt.figure()
    plt.imshow(grid, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            plt.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center")
    plt.tight_layout()
    plt.show()

# ------------------------- Main -------------------------
def main():
    env, nS, nA, P = make_env()

    # ——— Safety: Stelle sicher, dass es wirklich eine Gymnasium-Env ist
    assert isinstance(env, Env), f"env ist {type(env)} – erwartete gymnasium.Env"

    gamma = 0.99

    # "Wahrer" Reward nur für den Experten
    chars = grid_chars(env)
    R_true = np.zeros(nS, dtype=float)
    for s, ch in enumerate(chars):
        if ch == 'H': R_true[s] = -1.0
        elif ch == 'G': R_true[s] =  1.0
        else:           R_true[s] = -0.01

    # Expertenpolicy (für Demos)
    V_star, pi_star = value_iteration(P, nS, nA, R_true, gamma=gamma)
    print("Expertenpolicy (0=L,1=D,2=R,3=U):")
    print(pi_star.reshape(4,4))

    # Demos erzeugen
    N_TRAJ = 5000
    expert_trajs = [rollout_policy(env, pi_star, max_steps=100) for _ in range(N_TRAJ)]

    # One-Hot-Features
    Phi = np.eye(nS, dtype=float)

    # Feature-Expectations (Experte)
    mu_E = feature_expectations_from_demos(expert_trajs, Phi, gamma=gamma)
    print("||mu_E||:", np.linalg.norm(mu_E))

    # Max-Margin Apprenticeship Learning
    result = apprenticeship_learning_maxmargin(
        P, nS, nA, Phi, mu_E, gamma=gamma,
        start_state=0, max_iters=500, tol=1e-3, verbose=True
    )

    lam = result["lambdas"]
    pi_list = result["policies"]
    mu_list = result["mus"]
    mu_proj = result["mu_proj"]
    w_learned_dir = result["w"]
    gap = result["gap"]

    print("\n[ERGEBNIS] Abstand ||mu_E - mu_proj||:", gap)
    print("[ERGEBNIS] Mischungslambdas (über Policies):", lam)

    # Policy aus gelerntem Reward (R_w) mit der gelernten Richtungs- w
    R_learned = Phi @ w_learned_dir
    _, pi_learned = value_iteration(P, nS, nA, R_learned, gamma=gamma)
    print("Gelernte Policy (aus w-Richtung):")
    print(pi_learned.reshape(4,4))

    # Evaluation
    succ_expert  = eval_policy(env, pi_star, episodes=3000)
    succ_learned = eval_policy(env, pi_learned, episodes=3000)
    print(f"Erfolgsrate Experte : {succ_expert:.3f}")
    print(f"Erfolgsrate Gelernt : {succ_learned:.3f}")

    # Visuals
    plot_grid(R_true,    "Wahrer Reward (nur für Expert)")
    plot_grid(R_learned, "Gelearnte Reward-Richtung (w)")

    plt.figure()
    plt.bar(["||mu_E - mu_proj||"], [gap])
    plt.title("Max-Margin Lücke (final)")
    plt.tight_layout()
    plt.show()
    from matplotlib.collections import LineCollection

def _grid_size_from_states(nS: int) -> int:
    size = int(round(nS ** 0.5))
    assert size * size == nS, "nS muss ein Quadrat sein (z.B. 16 für 4x4)."
    return size

def _rc_from_state(s: int, size: int):
    r, c = divmod(s, size)
    return r, c

def _xy_center_of_cell(r: int, c: int):
    return float(c), float(r)

def _quiver_components_for_policy(pi, size: int):
    # Gym FrozenLake: 0=L, 1=D, 2=R, 3=U
    a2vec = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    X, Y, U, V = [], [], [], []
    for s, a in enumerate(pi):
        r, c = _rc_from_state(s, size)
        x, y = _xy_center_of_cell(r, c)
        dx, dy = a2vec[int(a)]
        X.append(x + 0.5)
        Y.append((size - 1 - y) + 0.5)  # invert y für „oben = größer“
        U.append(dx * 0.65)
        V.append(-dy * 0.65)            # minus, weil Plot-y nach oben
    return np.array(X), np.array(Y), np.array(U), np.array(V)

def _lines_from_trajectories(trajectories, size: int):
    lines = []
    for traj in trajectories:
        if not traj:
            continue
        pts = []
        for (s, a, s2) in traj:
            for st in (s, s2):
                r, c = _rc_from_state(st, size)
                x, y = _xy_center_of_cell(r, c)
                pts.append((x + 0.5, (size - 1 - y) + 0.5))
        # gleiche Punkte hintereinander entfernen
        dedup = [pts[0]]
        for p in pts[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        lines.append(np.array(dedup))
    return lines

def plot_policy_and_trajectories(pi, trajectories, bg_values=None, title="Policy & Trajectories"):
    """
    pi:      np.array(nS,) mit Aktionen {0:L,1:D,2:R,3:U}
    trajectories: Liste von Trajs; jede Traj = Liste von (s, a, s2)
    bg_values: optional np.array(nS,) -> Heatmap (z.B. Reward)
    """
    nS = len(pi)
    size = _grid_size_from_states(nS)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title)

    bg = np.zeros((size, size)) if bg_values is None else np.array(bg_values).reshape(size, size)
    im = ax.imshow(bg[::-1, :], interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Gitter
    for g in range(size + 1):
        ax.plot([0, size], [g, g], color="#711638", linewidth=1, alpha=0.6)
        ax.plot([g, g], [0, size], color="#711638", linewidth=1, alpha=0.6)

    # Policy-Pfeile
    X, Y, U, V = _quiver_components_for_policy(pi, size)
    ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1.0, width=0.012, alpha=0.9)

    # Trajektorien-Linien
    lines = _lines_from_trajectories(trajectories, size)
    if lines:
        lc = LineCollection(lines, colors="#c0c6cf", linewidths=3.0, alpha=0.9)
        ax.add_collection(lc)

    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    def main():
        # … dein Code …
        return {
            "pi_star": pi_star,
            "expert_trajs": expert_trajs,
            "R_true": R_true,
            "pi_learned": pi_learned,
            "R_learned": R_learned,
        }

   if __name__ == "__main__":
    out = main()
    plot_policy_and_trajectories(
        out["pi_star"], out["expert_trajs"][:3],
        bg_values=out["R_true"],
        title="Expert Trajectories and Policy (BG=true reward)"
    )
    plot_policy_and_trajectories(
        out["pi_learned"], out["expert_trajs"][:3],
        bg_values=out["R_learned"],
        title="Learned Policy + Expert Trajs (BG=learned reward)"
    )
