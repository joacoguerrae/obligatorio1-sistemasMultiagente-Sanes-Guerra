"""
Utilidades de experimentación para el Obligatorio 1 — Sistemas Multiagente (2026).
Contiene funciones reutilizables para ejecutar enfrentamientos, graficar evolución
de políticas y generar tablas de torneo round-robin.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from multiprocessing import Pool, cpu_count

try:
    from tqdm.notebook import tqdm
except ImportError:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        def tqdm(iterable, *args, **kwargs):
            return iterable

from agents.fictitiousplay import FictitiousPlay
from agents.regretmatching import RegretMatching
from agents.random_agent import RandomAgent
from agents.iql import IQL
from agents.jal import JAL

import os

# ─── Colores y estilo ────────────────────────────────────────────────────────

OUTPUT_DIR = None  # Setealo con setup_output_dir() para guardar figuras automáticamente

AGENT_COLORS = {
    "FP": "#2196F3",
    "RM": "#E91E63",
    "IQL": "#4CAF50",
    "JAL": "#FF9800",
    "Random": "#9E9E9E",
}

AGENT_FULL_NAMES = {
    "FP": "Fictitious Play",
    "RM": "Regret Matching",
    "IQL": "Independent Q-Learning",
    "JAL": "JAL – Agent Modeling",
    "Random": "Random Agent",
}


def setup_plot_style():
    """Configura el estilo global de matplotlib para el notebook."""
    plt.rcParams.update(
        {
            "figure.figsize": (13, 5),
            "font.size": 11,
            "font.family": "sans-serif",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 100,
            "savefig.dpi": 150,
        }
    )
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")


def setup_output_dir(path):
    """Setea la carpeta donde se guardan automáticamente las figuras generadas."""
    global OUTPUT_DIR
    OUTPUT_DIR = path
    os.makedirs(path, exist_ok=True)
    print(f'📁 Figuras se guardan en: {os.path.abspath(path)}')


def _save_fig(fig, name):
    """Guarda la figura si OUTPUT_DIR está configurado."""
    if OUTPUT_DIR is not None:
        # Sanitizar nombre de archivo
        safe_name = name.replace(' ', '_').replace('—', '-').replace(':', '')
        path = os.path.join(OUTPUT_DIR, f'{safe_name}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'  💾 {path}')


# ─── Factory de agentes ──────────────────────────────────────────────────────


def make_agent(name, game, agent_id, seed=None):
    """Instancia un agente por nombre corto."""
    constructors = {
        "FP": lambda: FictitiousPlay(game=game, agent=agent_id, seed=seed),
        "RM": lambda: RegretMatching(game=game, agent=agent_id, seed=seed),
        "IQL": lambda: IQL(
            game=game,
            agent=agent_id,
            alpha=0.1,
            epsilon=0.2,
            min_epsilon=0.01,
            epsilon_decay=0.9999,
            seed=seed,
        ),
        "JAL": lambda: JAL(
            game=game,
            agent=agent_id,
            alpha=0.1,
            epsilon=0.2,
            min_epsilon=0.01,
            epsilon_decay=0.9999,
            seed=seed,
        ),
        "Random": lambda: RandomAgent(game=game, agent=agent_id),
    }
    if name not in constructors:
        raise ValueError(f"Agente desconocido: {name}")
    return constructors[name]()


# ─── Ejecución de experimentos ───────────────────────────────────────────────


def run_experiment(
    game_class,
    agent_names,
    n_episodes=10000,
    track_interval=100,
    seed=42,
    **game_kwargs,
):
    """Ejecuta un enfrentamiento y trackea políticas y rewards a lo largo del tiempo."""
    game_name = game_class.__name__
    print(f'\n🎮 {agent_names[0]} vs {agent_names[1]} — {game_name} — {n_episodes:,} episodios')

    game = game_class(**game_kwargs)
    game.reset()

    agent_ids = game.agents
    agents = {
        agent_ids[i]: make_agent(agent_names[i], game, agent_ids[i], seed=seed + i)
        for i in range(len(agent_names))
    }

    rewards_acum = {a: 0.0 for a in agent_ids}
    policy_history = {a: [] for a in agent_ids}
    reward_history = {a: [] for a in agent_ids}

    for ep in tqdm(
        range(1, n_episodes + 1),
        desc=f"{agent_names[0]} vs {agent_names[1]}",
        unit="ep",
        leave=False,
    ):
        actions = {a: agents[a].action() for a in agent_ids}
        game.step(actions)
        for a in agent_ids:
            rewards_acum[a] += game.reward(a)

        if ep % track_interval == 0:
            for a in agent_ids:
                policy_history[a].append(agents[a].policy().copy())
                reward_history[a].append(rewards_acum[a] / ep)

    return {
        "agents": agents,
        "agent_names": agent_names,
        "agent_ids": agent_ids,
        "rewards_acum": rewards_acum,
        "policy_history": policy_history,
        "reward_history": reward_history,
        "n_episodes": n_episodes,
        "track_interval": track_interval,
    }


# ─── Torneo round-robin ─────────────────────────────────────────────────────

def _run_single_matchup(args):
    """Worker para un matchup individual (top-level para que sea picklable)."""
    game_class, agent_name_i, agent_name_j, n_episodes, seed, game_kwargs = args
    game = game_class(**game_kwargs)
    game.reset()
    ids = game.agents
    agents = {
        ids[0]: make_agent(agent_name_i, game, ids[0], seed=seed),
        ids[1]: make_agent(agent_name_j, game, ids[1], seed=seed + 1),
    }
    total_r = 0.0
    for _ in range(n_episodes):
        actions = {a: agents[a].action() for a in ids}
        game.step(actions)
        total_r += game.reward(ids[0])
    return total_r / n_episodes


def run_tournament(game_class, agent_list, n_episodes=10000, seed=42, workers=None, **game_kwargs):
    """Ejecuta torneo round-robin, paralelizado automáticamente según CPUs disponibles.

    Args:
        workers: número de procesos paralelos.
                 None → auto-detecta cpu_count().
                 1 → secuencial (sin multiprocessing).
    """
    if workers is None:
        workers = cpu_count()

    n = len(agent_list)
    total_matchups = n * n
    game_name = game_class.__name__
    mode = f'paralelo ({workers} CPUs)' if workers > 1 else 'secuencial'
    print(f'\n🏆 Torneo {game_name} — {n} agentes, {total_matchups} matchups '
          f'× {n_episodes:,} eps c/u — modo {mode}')

    matrix = np.zeros((n, n))
    matchup_args = [
        (game_class, agent_list[i], agent_list[j], n_episodes, seed, game_kwargs)
        for i in range(n) for j in range(n)
    ]

    if workers > 1:
        with Pool(workers) as pool:
            results = list(tqdm(
                pool.imap(_run_single_matchup, matchup_args),
                total=len(matchup_args),
                desc=f'Torneo {game_name} ({workers} CPUs)',
                leave=False,
            ))
        for idx, avg in enumerate(results):
            i, j = divmod(idx, n)
            matrix[i, j] = avg
    else:
        pbar = tqdm(matchup_args, desc=f'Torneo {game_name}', leave=False)
        for idx, args in enumerate(pbar):
            i, j = divmod(idx, n)
            pbar.set_postfix_str(f'{agent_list[i]} vs {agent_list[j]}')
            matrix[i, j] = _run_single_matchup(args)

    print(f'✅ Torneo {game_name} completo')
    return matrix


# ─── Visualizaciones ─────────────────────────────────────────────────────────


def plot_tournament_heatmap(matrix, agent_list, game_name):
    """Muestra la matriz de torneo como heatmap coloreado."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    n = len(agent_list)

    # Normalizamos el color con centro en 0
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agent_list, fontsize=11)
    ax.set_yticklabels(agent_list, fontsize=11)
    ax.set_xlabel("Columna (Oponente)", fontsize=11)
    ax.set_ylabel("Fila (Agente evaluado)", fontsize=11)
    ax.set_title(
        f"Torneo Round-Robin — {game_name}", fontsize=14, fontweight="bold", pad=12
    )

    # Texto en cada celda
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    fig.colorbar(im, ax=ax, label="Reward promedio", shrink=0.8)
    plt.tight_layout()
    _save_fig(fig, f'torneo_{game_name}')
    plt.show()


def plot_policy_evolution(result, game_name, action_labels=None):
    """Gráfica de evolución de políticas aprendidas (lado a lado)."""
    n_agents = len(result["agent_ids"])
    fig, axes = plt.subplots(1, n_agents, figsize=(7 * n_agents, 5))
    if n_agents == 1:
        axes = [axes]

    for idx, agent_id in enumerate(result["agent_ids"]):
        ax = axes[idx]
        history = np.array(result["policy_history"][agent_id])
        if len(history) == 0:
            continue
        x = np.arange(1, len(history) + 1) * result["track_interval"]
        n_actions = history.shape[1]
        labels = (
            action_labels
            if action_labels
            else [f"Acción {i}" for i in range(n_actions)]
        )

        for a_idx in range(n_actions):
            ax.plot(x, history[:, a_idx], label=labels[a_idx], linewidth=2, alpha=0.85)

        name = result["agent_names"][idx]
        ax.set_title(
            f"{AGENT_FULL_NAMES.get(name, name)}", fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Episodios")
        ax.set_ylabel("π (probabilidad)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Evolución de Políticas — {game_name}: "
        f"{result['agent_names'][0]} vs {result['agent_names'][1]}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    _save_fig(fig, f'politicas_{game_name}_{result["agent_names"][0]}_vs_{result["agent_names"][1]}')
    plt.show()


def plot_reward_evolution(result, game_name):
    """Gráfica de evolución del reward promedio acumulado."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for idx, agent_id in enumerate(result["agent_ids"]):
        history = result["reward_history"][agent_id]
        if len(history) == 0:
            continue
        x = np.arange(1, len(history) + 1) * result["track_interval"]
        name = result["agent_names"][idx]
        color = AGENT_COLORS.get(name, None)
        ax.plot(
            x,
            history,
            label=f"{AGENT_FULL_NAMES.get(name, name)}",
            linewidth=2,
            color=color,
            alpha=0.85,
        )

    ax.set_title(
        f"Reward Promedio Acumulado — {game_name}", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Episodios")
    ax.set_ylabel("Reward promedio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(fig, f'reward_{game_name}')
    plt.show()


def print_final_policies(result):
    """Imprime las políticas finales aprendidas por cada agente."""
    print("Políticas finales aprendidas:")
    for idx, agent_id in enumerate(result["agent_ids"]):
        name = result["agent_names"][idx]
        pi = result["agents"][agent_id].policy()
        pi_str = ", ".join(f"{p:.4f}" for p in pi)
        print(f"  {AGENT_FULL_NAMES.get(name, name):>25}:  [{pi_str}]")
    print()


def print_ranking(matrix, agent_list, game_name):
    """Imprime ranking de agentes por reward promedio en un torneo."""
    avg = matrix.mean(axis=1)
    ranking = sorted(zip(agent_list, avg), key=lambda x: -x[1])
    print(f"  Ranking en {game_name}:")
    for rank, (name, reward) in enumerate(ranking, 1):
        bar = "█" * int(max(0, (reward + 0.5) * 15))
        full = AGENT_FULL_NAMES.get(name, name)
        print(f"    {rank}. {full:>25}: {reward:>8.4f}  {bar}")
    print()
