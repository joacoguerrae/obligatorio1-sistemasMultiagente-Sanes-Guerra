import numpy as np
from itertools import product
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


class JAL(Agent):
    """Joint-Action Learning with Agent Modeling (JAL-AM).

    Maintains Q-values over joint actions Q(s, a) and builds empirical
    models of opponent policies to compute expected action values.
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

        # Q-table over joint actions: q_table[state_key][joint_action_tuple] = float
        self.q_table: dict[tuple, dict[tuple, float]] = {}

        # Opponent action counts: opp_counts[state_key][agent_id][action] = count
        self.opp_counts: dict[tuple, dict[AgentID, np.ndarray]] = {}

        self.last_state = None
        self.last_action = None
        self.last_joint_actions = None  # Stored externally for multi-state games
        self.rng = np.random.default_rng(seed)
        self.learn = True

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

    def _get_q(self, state_key: tuple, joint_action: tuple) -> float:
        """Returns Q-value for (state, joint_action), defaulting to 0."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        return self.q_table[state_key].get(joint_action, 0.0)

    def _set_q(self, state_key: tuple, joint_action: tuple, value: float) -> None:
        """Sets Q-value for (state, joint_action)."""
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][joint_action] = value

    def _get_opp_policy(self, state_key: tuple, opp: AgentID) -> np.ndarray:
        """Returns the empirical probability distribution of opponent actions."""
        if state_key not in self.opp_counts:
            self.opp_counts[state_key] = {}
        if opp not in self.opp_counts[state_key]:
            n_actions = self.game.num_actions(opp)
            self.opp_counts[state_key][opp] = np.ones(n_actions)  # Laplace smoothing
        counts = self.opp_counts[state_key][opp]
        return counts / np.sum(counts)

    def _update_opp_model(self, state_key: tuple, actions: dict) -> None:
        """Updates opponent action counts based on observed joint action."""
        if state_key not in self.opp_counts:
            self.opp_counts[state_key] = {}
        for opp in self.game.agents:
            if opp == self.agent:
                continue
            if opp not in self.opp_counts[state_key]:
                n_actions = self.game.num_actions(opp)
                self.opp_counts[state_key][opp] = np.ones(n_actions)
            self.opp_counts[state_key][opp][actions[opp]] += 1

    def _average_value(self, state_key: tuple, my_action: int) -> float:
        """Computes AV_i(s, a_i) = sum_{a_{-i}} Q(s, (a_i, a_{-i})) * pi_{-i}(a_{-i} | s)."""
        opponents = [a for a in self.game.agents if a != self.agent]
        opp_action_lists = [
            list(self.game.action_iter(opp)) for opp in opponents
        ]

        av = 0.0
        for opp_actions in product(*opp_action_lists):
            # Build the full joint action tuple
            joint = [0] * len(self.game.agents)
            joint[self.game.agent_name_mapping[self.agent]] = my_action
            for opp, opp_a in zip(opponents, opp_actions):
                joint[self.game.agent_name_mapping[opp]] = opp_a
            joint_tuple = tuple(joint)

            # Compute probability of this opponent action combo (independence assumption)
            prob = 1.0
            for opp, opp_a in zip(opponents, opp_actions):
                pi = self._get_opp_policy(state_key, opp)
                prob *= pi[opp_a]

            av += self._get_q(state_key, joint_tuple) * prob

        return av

    def set_last_actions(self, actions: dict) -> None:
        """Stores the last joint action dict (for multi-state games like Foraging
        where observe() returns state vectors instead of action dicts)."""
        self.last_joint_actions = actions

    def reset(self) -> None:
        """Resets episode tracking state."""
        self.last_state = None
        self.last_action = None
        self.last_joint_actions = None

    def update(self) -> None:
        """Performs Q-learning update on the joint-action Q-table."""
        if self.last_state is None or self.last_action is None or not self.learn:
            return

        obs = self.game.observe(self.agent)
        if obs is None:
            return

        reward = self.game.reward(self.agent)
        if reward is None:
            return

        new_state_key = self._get_state_key(obs)

        # Determine the joint actions: use stored actions if available,
        # otherwise try to use observation as action dict (matrix games)
        actions = self.last_joint_actions
        if actions is None:
            # In matrix games, observe() returns an ActionDict
            if isinstance(obs, dict):
                actions = obs
            else:
                return  # Cannot update without knowing joint actions

        # Build joint action tuple from observed actions
        joint_action = tuple(
            actions[a] for a in self.game.agents
        )

        # Update opponent model
        self._update_opp_model(self.last_state, actions)

        # Compute best AV in new state
        n_my_actions = self.game.num_actions(self.agent)
        best_av = max(
            self._average_value(new_state_key, a_i)
            for a_i in range(n_my_actions)
        )

        # Q-learning update on joint action
        old_q = self._get_q(self.last_state, joint_action)
        target = reward + self.gamma * best_av
        new_q = old_q + self.alpha * (target - old_q)
        self._set_q(self.last_state, joint_action, new_q)

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def action(self) -> int:
        """Updates internal state and chooses an action epsilon-greedily based on AV."""
        self.update()

        obs = self.game.observe(self.agent)
        state_key = self._get_state_key(obs)

        n_actions = self.game.num_actions(self.agent)

        if self.rng.random() < self.epsilon and self.learn:
            a = self.rng.choice(n_actions)
        else:
            av_values = np.array([
                self._average_value(state_key, a_i)
                for a_i in range(n_actions)
            ])
            max_av = np.max(av_values)
            best_actions = np.where(av_values == max_av)[0]
            a = self.rng.choice(best_actions)

        self.last_state = state_key
        self.last_action = a
        return int(a)

    def policy(self) -> np.ndarray:
        """Returns the current greedy policy distribution based on AV values."""
        obs = self.game.observe(self.agent)
        state_key = self._get_state_key(obs)
        n_actions = self.game.num_actions(self.agent)

        av_values = np.array([
            self._average_value(state_key, a_i)
            for a_i in range(n_actions)
        ])

        pi = np.full(n_actions, self.epsilon / n_actions)
        max_av = np.max(av_values)
        best_actions = np.where(av_values == max_av)[0]
        pi[best_actions] += (1.0 - self.epsilon) / len(best_actions)
        return pi
