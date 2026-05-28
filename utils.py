"""
Funciones auxiliares para los experimentos del Obligatorio 1.

Acá juntamos lo que usamos en varios notebooks: armar agentes, correr enfrentamientos
de a dos, hacer torneos round-robin y graficar la evolución de políticas y rewards.
La idea es no repetir el mismo boilerplate en cada notebook.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm.auto import tqdm

from agents.fictitiousplay import FictitiousPlay
from agents.regretmatching import RegretMatching
from agents.random_agent import RandomAgent
from agents.iql import IQL
from agents.jal import JAL

# ─── Colores y estilo ────────────────────────────────────────────────────────

# Colores fijos por agente para que las gráficas sean comparables entre notebooks.
AGENT_COLORS = {
    'FP': '#2196F3',
    'RM': '#E91E63',
    'IQL': '#4CAF50',
    'JAL': '#FF9800',
    'Random': '#9E9E9E',
}

AGENT_FULL_NAMES = {
    'FP': 'Fictitious Play',
    'RM': 'Regret Matching',
    'IQL': 'Independent Q-Learning',
    'JAL': 'JAL – Agent Modeling',
    'Random': 'Random Agent',
}


def setup_plot_style():
    """Estilo por defecto para las gráficas del informe."""
    plt.rcParams.update({
        'figure.figsize': (13, 5),
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'figure.dpi': 100,
        'savefig.dpi': 150,
    })
    # En algunas versiones de matplotlib no está seaborn-v0_8, caemos a ggplot.
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')


# ─── Factory de agentes ──────────────────────────────────────────────────────

def make_agent(name, game, agent_id, seed=None):
    """Devuelve una instancia del agente pedido a partir de su nombre corto.

    Centralizamos acá los hiperparámetros de IQL y JAL para que sean los mismos
    en todos los experimentos y los resultados sean comparables.
    """
    constructors = {
        'FP': lambda: FictitiousPlay(game=game, agent=agent_id, seed=seed),
        'RM': lambda: RegretMatching(game=game, agent=agent_id, seed=seed),
        'IQL': lambda: IQL(game=game, agent=agent_id,
                           alpha=0.1, epsilon=0.2, min_epsilon=0.01,
                           epsilon_decay=0.9999, seed=seed),
        'JAL': lambda: JAL(game=game, agent=agent_id,
                           alpha=0.1, epsilon=0.2, min_epsilon=0.01,
                           epsilon_decay=0.9999, seed=seed),
        'Random': lambda: RandomAgent(game=game, agent=agent_id),
    }
    if name not in constructors:
        raise ValueError(f"Agente desconocido: {name}")
    return constructors[name]()


# ─── Ejecución de experimentos ───────────────────────────────────────────────

def run_experiment(game_class, agent_names, n_episodes=10000,
                   track_interval=100, seed=42, **game_kwargs):
    """Corre un enfrentamiento entre dos agentes y guarda la evolución de cada uno.

    Cada `track_interval` episodios snapshoteamos la política y el reward promedio
    de cada agente — así después podemos graficar cómo fueron convergiendo.
    """
    game = game_class(**game_kwargs)
    game.reset()

    # Usamos los ids tal como los devuelve el juego (no asumimos "agent_0", "agent_1").
    agent_ids = game.agents
    agents = {
        agent_ids[i]: make_agent(agent_names[i], game, agent_ids[i], seed=seed + i)
        for i in range(len(agent_names))
    }

    rewards_acum = {a: 0.0 for a in agent_ids}
    policy_history = {a: [] for a in agent_ids}
    reward_history = {a: [] for a in agent_ids}

    for ep in tqdm(range(1, n_episodes + 1),
                    desc=f'{agent_names[0]} vs {agent_names[1]}',
                    leave=False):
        # Cada agente decide en simultáneo (no hay turnos, son juegos normales/estocásticos).
        actions = {a: agents[a].action() for a in agent_ids}
        game.step(actions)
        for a in agent_ids:
            rewards_acum[a] += game.reward(a)

        # Trackeo periódico para no saturar memoria.
        if ep % track_interval == 0:
            for a in agent_ids:
                policy_history[a].append(agents[a].policy().copy())
                reward_history[a].append(rewards_acum[a] / ep)

    return {
        'agents': agents,
        'agent_names': agent_names,
        'agent_ids': agent_ids,
        'rewards_acum': rewards_acum,
        'policy_history': policy_history,
        'reward_history': reward_history,
        'n_episodes': n_episodes,
        'track_interval': track_interval,
    }


# ─── Torneo round-robin ─────────────────────────────────────────────────────

def run_tournament(game_class, agent_list, n_episodes=10000,
                   seed=42, **game_kwargs):
    """Torneo round-robin: cada agente contra cada agente (incluido contra sí mismo).

    Devuelve la matriz n×n donde la celda (i,j) es el reward promedio del agente i
    (fila) cuando juega contra el j (columna).
    """
    n = len(agent_list)
    matrix = np.zeros((n, n))

    matchups = [(i, j) for i in range(n) for j in range(n)]
    pbar = tqdm(matchups, desc='Torneo round-robin', leave=False)
    for i, j in pbar:
        pbar.set_postfix_str(f'{agent_list[i]} vs {agent_list[j]}')
        game = game_class(**game_kwargs)
        game.reset()
        ids = game.agents
        agents = {
            ids[0]: make_agent(agent_list[i], game, ids[0], seed=seed),
            ids[1]: make_agent(agent_list[j], game, ids[1], seed=seed + 1),
        }
        total_r = 0.0
        for _ in range(n_episodes):
            actions = {a: agents[a].action() for a in ids}
            game.step(actions)
            # Reportamos solo el reward del agente "fila" — la matriz se construye así.
            total_r += game.reward(ids[0])
        matrix[i, j] = total_r / n_episodes

    return matrix


# ─── Visualizaciones ─────────────────────────────────────────────────────────

def plot_tournament_heatmap(matrix, agent_list, game_name):
    """Heatmap de la matriz del torneo, con verde = gana fila y rojo = pierde fila."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    n = len(agent_list)

    # Centramos la escala de color en 0 para que el blanco quede en "empate".
    # El max(..., 0.01) es para evitar vmin == vmax cuando la matriz es toda cero.
    vmax = max(abs(matrix.min()), abs(matrix.max()), 0.01)
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='equal')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agent_list, fontsize=11)
    ax.set_yticklabels(agent_list, fontsize=11)
    ax.set_xlabel('Columna (Oponente)', fontsize=11)
    ax.set_ylabel('Fila (Agente evaluado)', fontsize=11)
    ax.set_title(f'Torneo Round-Robin — {game_name}', fontsize=14, fontweight='bold', pad=12)

    # Escribimos el valor numérico en cada celda. Cambiamos a texto blanco cuando
    # el fondo es muy oscuro, si no no se lee.
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            color = 'white' if abs(val) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    fig.colorbar(im, ax=ax, label='Reward promedio', shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_policy_evolution(result, game_name, action_labels=None):
    """Cómo va cambiando la política de cada agente a lo largo del entrenamiento.

    Un subplot por agente. Si pasás `action_labels` se usan esos nombres en la leyenda
    (por ejemplo ["Piedra", "Papel", "Tijera"]), si no, "Acción 0", "Acción 1", etc.
    """
    n_agents = len(result['agent_ids'])
    fig, axes = plt.subplots(1, n_agents, figsize=(7 * n_agents, 5))
    # Si hay un solo agente, axes no viene como lista — lo envolvemos.
    if n_agents == 1:
        axes = [axes]

    for idx, agent_id in enumerate(result['agent_ids']):
        ax = axes[idx]
        history = np.array(result['policy_history'][agent_id])
        if len(history) == 0:
            continue
        # Convertimos índice de snapshot a número de episodio.
        x = np.arange(1, len(history) + 1) * result['track_interval']
        n_actions = history.shape[1]
        labels = action_labels if action_labels else [f'Acción {i}' for i in range(n_actions)]

        for a_idx in range(n_actions):
            ax.plot(x, history[:, a_idx], label=labels[a_idx], linewidth=2, alpha=0.85)

        name = result['agent_names'][idx]
        ax.set_title(f'{AGENT_FULL_NAMES.get(name, name)}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episodios')
        ax.set_ylabel('π (probabilidad)')
        # Margen de 0.05 para que las líneas en 0 o 1 no queden pegadas al borde.
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Evolución de Políticas — {game_name}: '
                 f'{result["agent_names"][0]} vs {result["agent_names"][1]}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_reward_evolution(result, game_name):
    """Reward promedio acumulado (running average) de cada agente."""
    fig, ax = plt.subplots(figsize=(10, 4))

    for idx, agent_id in enumerate(result['agent_ids']):
        history = result['reward_history'][agent_id]
        if len(history) == 0:
            continue
        x = np.arange(1, len(history) + 1) * result['track_interval']
        name = result['agent_names'][idx]
        color = AGENT_COLORS.get(name, None)
        ax.plot(x, history, label=f'{AGENT_FULL_NAMES.get(name, name)}',
                linewidth=2, color=color, alpha=0.85)

    ax.set_title(f'Reward Promedio Acumulado — {game_name}',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Episodios')
    ax.set_ylabel('Reward promedio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_final_policies(result):
    """Tira por consola la política final de cada agente — útil para chequear el equilibrio."""
    print("Políticas finales aprendidas:")
    for idx, agent_id in enumerate(result['agent_ids']):
        name = result['agent_names'][idx]
        pi = result['agents'][agent_id].policy()
        pi_str = ', '.join(f'{p:.4f}' for p in pi)
        print(f"  {AGENT_FULL_NAMES.get(name, name):>25}:  [{pi_str}]")
    print()


def print_ranking(matrix, agent_list, game_name):
    """Ranking de agentes según el reward promedio sobre toda su fila del torneo."""
    # Promediamos la fila: cuánto saca el agente i en promedio contra todos.
    avg = matrix.mean(axis=1)
    ranking = sorted(zip(agent_list, avg), key=lambda x: -x[1])
    print(f"  Ranking en {game_name}:")
    for rank, (name, reward) in enumerate(ranking, 1):
        # Barrita visual. Sumamos 0.5 para que rewards negativos chicos no rompan.
        bar = '█' * int(max(0, (reward + 0.5) * 15))
        full = AGENT_FULL_NAMES.get(name, name)
        print(f"    {rank}. {full:>25}: {reward:>8.4f}  {bar}")
    print()
