from itertools import product
import numpy as np
from numpy import ndarray
from base.agent import Agent
from base.game import SimultaneousGame, AgentID


class FictitiousPlay(Agent):
    def __init__(
        self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None
    ) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)

        self.count: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar count con initial si no es None o, caso contrario, con valores random
        #
        for agent in self.game.agents:
            self.count[agent] = (
                initial[agent].copy()
                if initial is not None
                else np.ones(self.game.num_actions(agent))
            )

        self.learned_policy: dict[AgentID, ndarray] = {}
        #
        # TODO: inicializar learned_policy usando de count
        #
        for agent in self.game.agents:
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def get_rewards(self) -> dict:
        if hasattr(self, '_cached_rewards'):
            return self._cached_rewards
            
        g = self.game.clone()
        agents_actions = list(map(lambda agent: list(g.action_iter(agent)), g.agents))
        rewards: dict[tuple, float] = {}
        #
        # TODO: calcular los rewards de agente para cada acción conjunta
        # Ayuda: usar product(*agents_actions) de itertools para iterar sobre agents_actions
        # s
        # OBTENEMOS TODAS LAS COMBINACIONES POSIBLES
        combinaciones = product(*agents_actions)
        # Iteramos sobre las combinaciones y obtenemos el reward
        for joint_action in combinaciones:
            action_dict = dict(zip(g.agents, joint_action))
            g.reset()
            g.step(action_dict)
            rewards[joint_action] = g.reward(self.agent)

        self._cached_rewards = rewards
        return rewards

    def get_utility(self):
        rewards = self.get_rewards()
        utility = np.zeros(self.game.num_actions(self.agent))
        #
        # TODO: calcular la utilidad (valor) de cada acción de agente.
        # Ayuda: iterar sobre rewards para cada acción de agente
        #
        for joint_action, reward in rewards.items():
            prob = 1.0
            for agent in self.game.agents:
                if agent != self.agent:
                    action = joint_action[self.game.agent_name_mapping[agent]]
                    prob *= self.learned_policy[agent][action]
            a = joint_action[self.game.agent_name_mapping[self.agent]]
            utility[a] += reward * prob
        return utility

    def bestresponse(self):
        a = np.argmax(self.get_utility())
        #
        # TODO: retornar la acción de mayor utilidad
        #
        return a

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        for agent in self.game.agents:
            self.count[agent][actions[agent]] += 1
            self.learned_policy[agent] = self.count[agent] / np.sum(self.count[agent])

    def action(self):
        self.update()
        return self.bestresponse()

    def policy(self):
        return self.learned_policy[self.agent]
