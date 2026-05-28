import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


class IQL(Agent):
    """Independent Q-Learning (IQL) agent.
    Learns state-action values Q(s, a_i) independently, treating other agents as part of the environment.
    """
    def __init__(
        self,
        game: SimultaneousGame,
        agent: AgentID,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__(game=game, agent=agent)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table: dict[tuple, np.ndarray] = {}
        self.last_state = None
        self.last_action = None
        self.rng = np.random.default_rng(seed)
        self.learn = True  # Can be disabled for evaluation

    def _get_state_key(self, obs) -> tuple:
        """Converts observation into a hashable state key."""
        if obs is None:
            return (0,)
        if isinstance(obs, np.ndarray):
            return tuple(obs.flatten())
        if isinstance(obs, dict):
            return tuple(sorted((k, v) for k, v in obs.items() if v is not None))
        if isinstance(obs, (list, tuple)):
            return tuple(obs)
        return (obs,)

    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Returns the Q-values array for the given state, initializing it to zeros if not present."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.game.num_actions(self.agent))
        return self.q_table[state_key]

    def reset(self) -> None:
        """Resets the history tracking at the start of a new episode."""
        self.last_state = None
        self.last_action = None

    def update(self) -> None:
        """Performs the Q-learning update using the transition from the previous round."""
        if self.last_state is None or self.last_action is None or not self.learn:
            return

        # Observe new state and reward
        new_obs = self.game.observe(self.agent)
        new_state_key = self._get_state_key(new_obs)
        reward = self.game.reward(self.agent)

        if reward is None:
            return

        # Perform standard Q-learning update
        q_values_old = self._get_q_values(self.last_state)
        q_values_new = self._get_q_values(new_state_key)

        target = reward + self.gamma * np.max(q_values_new)
        q_values_old[self.last_action] += self.alpha * (target - q_values_old[self.last_action])

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def action(self) -> int:
        """Updates internal state and chooses an action epsilon-greedily."""
        # Perform learning update from the previous transition
        self.update()

        # Observe current state
        obs = self.game.observe(self.agent)
        state_key = self._get_state_key(obs)
        q_values = self._get_q_values(state_key)

        # Epsilon-greedy action selection
        if self.rng.random() < self.epsilon and self.learn:
            a = self.rng.choice(self.game.num_actions(self.agent))
        else:
            # Argmax with random break for ties
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            a = self.rng.choice(best_actions)

        # Track history for the next step
        self.last_state = state_key
        self.last_action = a
        return int(a)

    def policy(self) -> np.ndarray:
        """Returns the current exploration policy distribution over actions for the current state."""
        obs = self.game.observe(self.agent)
        state_key = self._get_state_key(obs)
        q_values = self._get_q_values(state_key)

        num_actions = self.game.num_actions(self.agent)
        pi = np.full(num_actions, self.epsilon / num_actions)

        max_q = np.max(q_values)
        best_actions = np.where(q_values == max_q)[0]
        pi[best_actions] += (1.0 - self.epsilon) / len(best_actions)
        return pi
