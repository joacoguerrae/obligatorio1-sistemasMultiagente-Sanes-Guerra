import numpy as np
from numpy import ndarray
from gymnasium.spaces import Discrete
from base.game import SimultaneousGame, ActionDict, ObsDict, AgentID

# Reward matrix configurations
# Each configuration contains:
#   - 'rewards': 4D array [agent_0_action][agent_1_action][agent_2_action][agent_rewards]
#   - 'num_actions': list of number of actions for each player [agent_0, agent_1, agent_2]
REWARD_MATRICES = {
    # G. Bonanno. Game Theory. Figure 2.22: A three-player game with ordinal payoffs
    1: {
        'rewards': np.array([
            [
                [
                    [0., 0., 0.],
                    [0., 0., 0.]
                ],
                [
                    [2., 8., 6.],
                    [1., 2., 5.]
                ]
            ],
            [
                [
                    [5., 3., 2.],
                    [1., 6., 1.]
                ],
                [
                    [3., 4., 2.],
                    [0., 0., 1.]
                ]
            ]
        ]),
        'num_actions': [2, 2, 2]
    },
    
    # R.J. Aumann, Randomized strategies. Example 2.3.
    2: {
        'rewards': np.array([
            [
                [
                    [0., 8., 0.],
                    [0., 0., 0.]
                ],
                [
                    [3., 3., 3.],
                    [3., 3., 3.]
                ]
            ],
            [
                [
                    [1., 1., 1.],
                    [1., 1., 1.]
                ],
                [
                    [0., 0., 0.],
                    [8., 0., 0.]
                ]
            ]
        ]),
        'num_actions': [2, 2, 2]
    },

    # R.J. Aumann, Randomized strategies. Example 2.5.
    # Shape: (2, 2, 3, 3) -> [agent_0: 2 actions][agent_1: 2 actions][agent_2: 3 actions][3 rewards]
    3: {
        'rewards': np.array([
            [  # agent_0 = T (action 0)
                [  # agent_1 = L (action 0)
                    [0., 0., 3.],  # agent_2 = W (action 0)
                    [2., 2., 2.],  # agent_2 = E (action 1)
                    [0., 0., 0.]   # agent_2 = N (action 2)
                ],
                [  # agent_1 = R (action 1)
                    [0., 0., 0.],  # agent_2 = W (action 0)
                    [0., 0., 0.],  # agent_2 = W (action 0)
                    [0., 0., 0.],  # agent_2 = W (action 0)
                ]
            ],
            [  # agent_0 = B (action 1)
                [  # agent_1 = L (action 0)
                    [1., 0., 0.],  # agent_2 = W (action 0)
                    [0., 0., 0.],  # agent_2 = W (action 0)
                    [0., 1., 0.],  # agent_2 = W (action 0)
                ],
                [  # agent_1 = R (action 1)
                    [0., 0., 0.],  # agent_2 = W (action 0)
                    [2., 2., 2.],  # agent_2 = W (action 0)
                    [0., 0., 3.],  # agent_2 = W (action 0)
                ]
            ]
        ]),
        'num_actions': [2, 2, 3]  # agent_0: 2, agent_1: 2, agent_2: 3
    },

    # R.J. Aumann, Randomized strategies. Example 2.6.
    # Shape: (2, 2, 3, 3) -> [agent_0: 2 actions][agent_1: 2 actions][agent_2: 3 actions][3 rewards]
    4: {
        'rewards': np.array([
            [  # agent_0 = T (action 0)
                [  # agent_1 = L (action 0)
                    [0., 1., 3.],  # agent_2 = W (action 0)
                    [2., 2., 2.],  # agent_2 = E (action 1)
                    [0., 1., 0.]   # agent_2 = N (action 2)
                ],
                [  # agent_1 = R (action 1)
                    [0., 0., 0.],  # agent_2 = W (action 0)
                    [0., 0., 0.],  # agent_2 = W (action 0)
                    [0., 0., 0.],  # agent_2 = W (action 0)
                ]
            ],
            [  # agent_0 = B (action 1)
                [  # agent_1 = L (action 0)
                    [1., 1., 1.],  # agent_2 = W (action 0)
                    [2., 2., 0.],  # agent_2 = W (action 0)
                    [1., 1., 1.],  # agent_2 = W (action 0)
                ],
                [  # agent_1 = R (action 1)
                    [1., 0., 0.],  # agent_2 = W (action 0)
                    [2., 2., 2.],  # agent_2 = W (action 0)
                    [1., 0., 3.],  # agent_2 = W (action 0)
                ]
            ]
        ]),
        'num_actions': [2, 2, 3]  # agent_0: 2, agent_1: 2, agent_2: 3
    }
}

class ThreePlayers(SimultaneousGame):

    def __init__(self, config=1):
        """
        Initialize ThreePlayers game.
        
        Args:
            config: Configuration to select reward matrix (default: 1)
                   Available configs: {', '.join(map(str, REWARD_MATRICES.keys()))}
        """
        # agents
        self.agents = ["agent_" + str(r) for r in range(3)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        # Store config
        self.config = config

        # Select reward matrix based on config
        if config not in REWARD_MATRICES:
            available_configs = ', '.join(map(str, sorted(REWARD_MATRICES.keys())))
            raise ValueError(f"Invalid config value: {config}. Available configs: {available_configs}")
        
        config_data = REWARD_MATRICES[config]
        self._R = config_data['rewards']
        self._num_actions_per_agent = config_data['num_actions']
        
        # Validate that reward matrix dimensions match num_actions
        expected_shape = tuple(self._num_actions_per_agent + [3])  # [a0, a1, a2, 3]
        if self._R.shape != expected_shape:
            raise ValueError(
                f"Config {config}: Reward matrix shape {self._R.shape} does not match "
                f"expected shape {expected_shape} based on num_actions {self._num_actions_per_agent}. "
                f"The shape should be [{self._num_actions_per_agent[0]}, {self._num_actions_per_agent[1]}, "
                f"{self._num_actions_per_agent[2]}, 3]."
            )

        # actions
        # Define action labels for each agent based on number of actions
        # This is a basic mapping; can be customized for specific games
        action_labels = [
            ['T', 'B', 'M'],  # agent_0: Top, Bottom, Middle
            ['L', 'R', 'C'],  # agent_1: Left, Right, Center
            ['W', 'E', 'N']   # agent_2: West, East, North (or other labels)
        ]
        
        self._moves = {}
        self.action_spaces = {}
        
        for i, agent in enumerate(self.agents):
            num_actions = self._num_actions_per_agent[i]
            self._moves[agent] = action_labels[i][:num_actions]
            self.action_spaces[agent] = Discrete(num_actions)

        # observations
        self.observation_spaces = {
            agent: ActionDict for agent in self.agents
        }

    def step(self, actions: ActionDict) -> tuple[ObsDict, dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict]]:
        # rewards
        (a0, a1, a2) = tuple(map(lambda agent: actions[agent], self.agents))
        for agent in self.agents:
            self.rewards[agent] = self._R[a0][a1][a2][self.agent_name_mapping[agent]]

        # observations
        self.observations = dict(map(lambda agent: (agent, actions), self.agents))

        # etcetera
        self.terminations = dict(map(lambda agent: (agent, True), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.observations = dict(map(lambda agent: (agent, None), self.agents))
        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def render(self) -> ndarray | str | list | None:
        for agent in self.agents:
            print(agent, self._moves[agent][self.observations[agent][agent]], self.rewards[agent])
    
    def get_config_info(self) -> dict:
        """
        Get information about the current configuration.
        
        Returns:
            dict: Configuration information including config number, number of actions per agent,
                  and action labels for each agent
        """
        return {
            'config': self.config,
            'num_actions_per_agent': self._num_actions_per_agent,
            'action_labels': {agent: self._moves[agent] for agent in self.agents},
            'reward_shape': self._R.shape
        }