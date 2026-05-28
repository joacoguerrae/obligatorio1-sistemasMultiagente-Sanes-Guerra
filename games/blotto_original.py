from itertools import product, filterfalse
import numpy as np
from numpy import ndarray
from gymnasium.spaces import Discrete
from base.game import SimultaneousGame, ActionDict, ObsDict, AgentID


def to_ord(s):
    return list(map(lambda x: ord(x) - 64, s))


class Blotto_Original(SimultaneousGame):
    def __init__(self, S=3, N=2):
        assert N > 1 and N <= S and S < 50
        self.S = S  # number of soldiers
        self.N = N  # number of battle fields

        # agents
        self.agents = ["agent_" + str(r) for r in range(2)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

        # actions
        self.set_moves()  # vectors (s_0, ..., s_{N-1})
        self._num_actions = len(self._moves)  # number of actions
        self.action_spaces = {  # actions are vector indices
            agent: Discrete(self._num_actions) for agent in self.agents
        }

        # observations
        self.observation_spaces = {agent: ActionDict for agent in self.agents}

        # utility matrix
        self.set_R()

    def set_moves(self):
        h = product(range(self.S + 1), repeat=self.N)
        f = filter(lambda l: sum(l) == self.S, h)
        self._moves = list(f)

    def _U(self, x, y):
        winx = np.sum(x > y)
        winy = np.sum(x < y)
        ux = 1 if winx > winy else (-1 if winx < winy else 0)
        return ux

    def set_R(self):
        self._R = np.zeros((self._num_actions, self._num_actions))
        for a in range(self._num_actions):
            for b in range(self._num_actions):
                self._R[a][b] = self._U(
                    np.array(self._moves[a]), np.array(self._moves[b])
                )

    def step(
        self, actions: ActionDict
    ) -> tuple[
        ObsDict,
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        # rewards
        (a0, a1) = tuple(map(lambda agent: actions[agent], self.agents))
        r = self._R[a0][a1]
        self.rewards[self.agents[0]] = r
        self.rewards[self.agents[1]] = -r

        # observations
        self.observations = dict(map(lambda agent: (agent, actions), self.agents))

        # etcetera
        self.terminations = dict(map(lambda agent: (agent, True), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

        return (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.observations = dict(map(lambda agent: (agent, None), self.agents))
        self.rewards = dict(map(lambda agent: (agent, None), self.agents))
        self.terminations = dict(map(lambda agent: (agent, False), self.agents))
        self.truncations = dict(map(lambda agent: (agent, False), self.agents))
        self.infos = dict(map(lambda agent: (agent, {}), self.agents))

    def render(self) -> ndarray | str | list | None:
        for agent in self.agents:
            print(
                agent, self._moves[self.observations[agent][agent]], self.rewards[agent]
            )
