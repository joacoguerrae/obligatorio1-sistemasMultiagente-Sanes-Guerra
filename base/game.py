import copy
from pettingzoo.utils.env import ParallelEnv, ObsDict, AgentID, ActionDict

class SimultaneousGame(ParallelEnv):

    observations: ObsDict
    rewards: dict[AgentID, float]
    terminations: dict[AgentID, bool]
    truncations: dict[AgentID, bool]
    infos: dict[AgentID, dict]

    agent_name_mapping: dict[AgentID, int]

    def observation_space(self, agent: AgentID):
        return self.observation_spaces[agent]

    def action_space(self, agent: AgentID):
        return self.action_spaces[agent]

    def num_actions(self, agent: AgentID):
        return self.action_space(agent).n
    
    def action_iter(self, agent: AgentID):
        return range(self.action_space(agent).start, self.action_space(agent).n)
        
    def observe(self, agent: AgentID):
        return self.observations[agent]
    
    def reward(self, agent: AgentID):
        return self.rewards[agent]
    
    def clone(self):
        game = copy.deepcopy(self)
        game.reset()
        return game





