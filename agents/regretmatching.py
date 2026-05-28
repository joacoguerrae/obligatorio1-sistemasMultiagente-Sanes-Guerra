import numpy as np
from base.agent import Agent
from base.game import SimultaneousGame, AgentID, ActionDict


class RegretMatching(Agent):
    def __init__(
        self, game: SimultaneousGame, agent: AgentID, initial=None, seed=None
    ) -> None:
        super().__init__(game=game, agent=agent)
        if initial is None:
            self.curr_policy = np.full(
                self.game.num_actions(self.agent), 1 / self.game.num_actions(self.agent)
            )
        else:
            self.curr_policy = initial.copy()
        self.cum_regrets = np.zeros(self.game.num_actions(self.agent))
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1
        np.random.seed(seed=seed)
        self._eval_game = game.clone()

    def regrets(self, played_actions: ActionDict) -> dict[AgentID, float]:
        actions = played_actions.copy()

        g = self._eval_game
        u = np.zeros(g.num_actions(self.agent), dtype=float)
        #
        # TODO: calcular regrets
        #
        g.reset()
        g.step(actions)
        real_reward = g.reward(self.agent)
        for act in g.action_iter(self.agent):
            actions[self.agent] = act
            g.reset()
            g.step(actions)
            u[act] = g.reward(self.agent)

        r = u - real_reward
        return r

    def regret_matching(self):
        #
        # TODO: calcular curr_policy y actualizar sum_policy
        #
        positive = np.maximum(self.cum_regrets, 0)
        total = np.sum(positive)
        if total > 0:
            self.curr_policy = positive / total
        else:
            self.curr_policy = np.full(len(self.cum_regrets), 1 / len(self.cum_regrets))
        self.sum_policy += self.curr_policy

    def update(self) -> None:
        actions = self.game.observe(self.agent)
        if actions is None:
            return
        regrets = self.regrets(actions)
        self.cum_regrets += regrets
        self.regret_matching()
        self.niter += 1
        self.learned_policy = self.sum_policy / self.niter

    def action(self):
        self.update()
        return np.argmax(np.random.multinomial(1, self.curr_policy, size=1))

    def policy(self):
        return self.learned_policy
