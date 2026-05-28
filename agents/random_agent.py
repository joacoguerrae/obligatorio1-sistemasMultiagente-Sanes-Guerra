import numpy as np
from base.game import SimultaneousGame, AgentID
from base.agent import Agent

class RandomAgent(Agent):

    def __init__(self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None) -> None:
        super().__init__(game=game, agent=agent)
        np.random.seed(seed=seed)
        if initial is None:
            self._policy = np.full(self.game.num_actions(self.agent), 1/self.game.num_actions(self.agent))
        else:
            self._policy = initial

    def action(self):
        actions = np.array(self.game.action_iter(self.agent))
        return np.random.choice(actions, size=1, p=self._policy)[0]
    
    def policy(self):
        return self._policy
    