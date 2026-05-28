from base.game import SimultaneousGame, AgentID

class Agent():

    def __init__(self, game:SimultaneousGame, agent: AgentID) -> None:
        self.game = game
        self.agent = agent

    def action(self):
        pass

    def policy(self):
        pass
    
