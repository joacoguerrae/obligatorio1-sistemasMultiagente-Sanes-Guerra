from base.game import SimultaneousGame, ActionDict
import gymnasium as gym
from lbforaging.foraging.environment import Action


class Foraging(SimultaneousGame):
    """Thin adapter from lbforaging Gymnasium env to project SimultaneousGame API.
    """

    def __init__(self, config: str | None = None, seed: int | None = None):
        if config is None:
            config = "Foraging-8x8-2p-1f-v3"

        self.env = gym.make(config)
        self.seed = seed

        self.action_set = [a.name for a in list(Action)]

        n_agents = self.env.unwrapped.n_agents
        self.agents = [f"agent_{i}" for i in range(n_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.agents)}

        self.action_spaces = self._build_per_agent_spaces(self.env.action_space, self.agents)

        self.observations = None
        self.rewards = None
        self.terminations = None
        self.truncations = None
        self.infos = None
        self.current_step = 0
        self._done = False
        self._truncated = False

    @staticmethod
    def _build_per_agent_spaces(space_obj, agents: list[str]) -> dict:
        if hasattr(space_obj, "spaces"):
            spaces = list(space_obj.spaces)
        elif isinstance(space_obj, (list, tuple)):
            spaces = list(space_obj)
        else:
            spaces = [space_obj for _ in agents]

        return {agent: spaces[i] for i, agent in enumerate(agents)}

    def num_agents(self):
        return len(self.agents)

    def _refresh_state(self, obs_tuple):
        self.observations = {
            agent: obs_tuple[self.agent_name_mapping[agent]].copy()
            for agent in self.agents
        }
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self._done = False
        self._truncated = False

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is None:
            seed = self.seed

        obs, info = self.env.reset(seed=seed, options=options)
        self.current_step = self.env.unwrapped.current_step
        self._refresh_state(obs)

        # ParallelEnv-style return: obs dict + infos dict.
        self.infos = {agent: info for agent in self.agents}
        return self.observations, self.infos

    def step(self, actions: ActionDict) -> tuple[dict, dict, dict, dict, dict]:
        joint_action = tuple(actions[agent] for agent in self.agents)
        obs, rewards, done, truncated, info = self.env.step(action=joint_action)

        self.current_step = self.env.unwrapped.current_step

        for i, agent in enumerate(self.agents):
            self.observations[agent] = obs[i].copy()
            self.rewards[agent] = rewards[i]
            self.terminations[agent] = bool(done)
            self.truncations[agent] = bool(truncated)
            self.infos[agent] = info

        self._done = bool(done)
        self._truncated = bool(truncated)

        return self.observations, self.rewards, self.terminations, self.truncations, self.infos

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def done(self):
        return self._done