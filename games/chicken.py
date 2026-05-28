import numpy as np
from numpy import ndarray
from gymnasium.spaces import Discrete
from base.game import SimultaneousGame, ActionDict, ObsDict, AgentID

class Chicken(SimultaneousGame):

    def __init__(self):
        self._R = { 
            'agent_0': np.array([[0., 7.], [2., 6.]]), 
            'agent_1': np.array([[0., 2.], [7., 6.]]) 
        }

        # agents
        self.agents = ["agent_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        # actions
        self._moves = ['S', 'L']
        self._num_actions = 2
        self.action_spaces = {
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # observations
        self.observation_spaces = {
            agent: ActionDict for agent in self.agents
        }

    def step(self, actions: ActionDict) -> tuple[ObsDict, dict[AgentID, float], dict[AgentID, bool], dict[AgentID, bool], dict[AgentID, dict]]:
        # rewards
        (a0, a1) = tuple(map(lambda agent: actions[agent], self.agents))
        for agent in self.agents:
            self.rewards[agent] = self._R[agent][a0][a1]

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
            print(agent, self._moves[self.observations[agent][agent]], self.rewards[agent])
