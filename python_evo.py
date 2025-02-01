import random


next_id = 0

class Agent:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
    

class Agentfactory:

    def __init__(self):
        self.agents = []
    
    def add_random_agent(self):
        global next_id
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        self.agents.append(Agent(next_id, x, y))
        next_id += 1

ecosystem = Agentfactory()

ecosystem.add_random_agent()

agent = ecosystem.agents[0]

print(agent.id)

