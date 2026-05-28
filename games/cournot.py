import numpy as np
from numpy import ndarray
from gymnasium.spaces import Discrete
from base.game import SimultaneousGame, ActionDict, ObsDict, AgentID

class Cournot(SimultaneousGame):
    """
    Cournot competition game with n firms.
    
    N firms compete by choosing quantities. Each firm i produces quantity qi.
    Price: P = a - b*Q where Q = sum(qi)
    Cost: C_i = c * qi
    Profit: π_i = P * qi - C_i = (a - b*Q) * qi - c * qi
    
    Parameters:
    - a: demand intercept (positive constant)
    - b: demand slope (positive constant)  
    - c: marginal cost (positive constant)
    - n_players: number of firms (default: 2)
    - max_quantity: maximum quantity each firm can produce
    - num_actions: number of discrete quantity levels (0 to max_quantity)
    """

    def __init__(self, a=10.0, b=1.0, c=2.0, n_players=2, max_quantity=10, num_actions=11):
        """
        Initialize Cournot game.
        
        Args:
            a: demand intercept (default: 10.0)
            b: demand slope (default: 1.0)
            c: marginal cost (default: 2.0)
            n_players: number of firms (default: 2)
            max_quantity: maximum quantity (default: 10)
            num_actions: number of discrete quantity levels (default: 11, giving 0,1,2,...,10)
        """
        self.a = a
        self.b = b
        self.c = c
        self.n_players = n_players
        self.max_quantity = max_quantity
        
        # Map discrete actions to actual quantities
        self._quantities = np.linspace(0, max_quantity, num_actions)
        
        # agents
        self.agents = ["agent_" + str(r) for r in range(n_players)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        # actions (discrete indices that map to quantities)
        self._num_actions = num_actions
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # observations
        self.observation_spaces = {
            agent: ActionDict for agent in self.agents
        }

    def _compute_profit(self, q_i, Q_total):
        """
        Compute profit for firm i given its quantity q_i and total market quantity Q_total.
        
        Profit = (a - b*Q_total) * q_i - c * q_i
        
        Args:
            q_i: quantity produced by firm i
            Q_total: total market quantity (sum of all firms)
        """
        P = self.a - self.b * Q_total  # price
        profit = P * q_i - self.c * q_i
        return profit

    def step(self, actions: ActionDict) -> tuple[ObsDict, dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict]]:
        """
        Execute one step of the game.
        
        Args:
            actions: Dictionary mapping agent names to action indices
            
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Convert action indices to quantities
        quantities = {agent: self._quantities[actions[agent]] for agent in self.agents}
        
        # Calculate total market quantity
        Q_total = sum(quantities.values())
        
        # Calculate rewards (profits) for each firm
        for agent in self.agents:
            self.rewards[agent] = self._compute_profit(quantities[agent], Q_total)

        # observations
        self.observations = dict(map(lambda agent: (agent, actions), self.agents))

        # etcetera
        self.terminations = dict(map(lambda agent: (agent, True), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the game to initial state."""
        self.observations = dict(map(lambda agent: (agent, None), self.agents))
        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def render(self) -> ndarray | str | list | None:
        """Display the current state of the game."""
        for agent in self.agents:
            if self.observations[agent] is not None:
                action_idx = self.observations[agent][agent]
                quantity = self._quantities[action_idx]
                print(f"{agent}: quantity={quantity:.2f}, profit={self.rewards[agent]:.2f}")
    
    def get_nash_equilibrium(self):
        """
        Calculate the Nash equilibrium for the Cournot game.
        
        For n symmetric firms with linear demand and constant marginal cost:
        q* = (a - c) / ((n+1)*b)
        Q* = n*q*
        P* = a - b*Q*
        π* = (P* - c)*q*
        
        Returns:
            tuple: (equilibrium quantity for each firm, equilibrium price, equilibrium profit for each firm)
        """
        q_star = (self.a - self.c) / ((self.n_players + 1) * self.b)
        Q_star = self.n_players * q_star
        P_star = self.a - self.b * Q_star
        profit_star = self._compute_profit(q_star, Q_star)
        
        return q_star, P_star, profit_star
