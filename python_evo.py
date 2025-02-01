import pygame
import random
import time
import math
import copy


next_id = 0
AGENT_COUNT = 1200
MAX_NODE_SPACE = 0.01
MAX_MAX_NODE_SPACE = .05
NODE_COUNT = 3
DELTA_T = .02
MAX_START_VELOCITY = 0.02
VELOCITY_DAMPENING = 0.93
CONSTANT_MASS = 0.5
WEIGHTS_COUNT = 500
WEIGHTS_SIGMA = 0.1
MAX_WEIGHT_MAG = 2
MEMORY_LENGTH = 3
METABOLIZE_CHANCE = 0.3
MAX_BIRTH_DIST = 0.02
FORCE_MULTIPLIER = .9



FOOD_TILES_ACROSS = 300
FOOD_GEN_RATE = 500

FRAMES_PER_SEC = 200
NODE_WIDTH = 4
LINE_WIDTH = 2

itC = 0

food_grid = []
for x in range(FOOD_TILES_ACROSS):
    temp = []
    for y in range(FOOD_TILES_ACROSS):
        temp.append(100)
    food_grid.append(temp)


pygame.init()

WIDTH = 1500
HEIGHT = WIDTH
screen = pygame.display.set_mode((WIDTH, HEIGHT))


class Agent:
    def __init__(self, id, node_states, node_locs, node_velocities, equilibrium_lengths, equilibrium_length_multipliers, weights, food, memory):
        self.id = id
        self.node_states = node_states
        self.birth_node_states = node_states
        self.node_locs = node_locs
        self.node_velocities = node_velocities
        self.birth_node_locs = node_locs
        self.equilibrium_lengths = equilibrium_lengths
        self.equilibrium_length_multipliers = equilibrium_length_multipliers
        self.weights = weights
        self.food = food
        self.memory = memory

    def eat(self):
        food_index_x = min(math.floor(self.node_locs[0][0] * FOOD_TILES_ACROSS), FOOD_TILES_ACROSS - 1)
        food_index_y = min(math.floor(self.node_locs[0][1] * FOOD_TILES_ACROSS), FOOD_TILES_ACROSS - 1)
        if food_grid[food_index_x][food_index_y] > 0:
            food_grid[food_index_x][food_index_y] -= 1
            self.food += 1
    
    def metabolize(self):
        self.food -= 1
        
        
    
    def think(self):
        k = 0
        nodes = []
        nodes.append(1)
        nodes.append((itC % 2) / 1)
        nodes.append((itC % 10) / 10)
        nodes.append((itC % 100) / 100)
        nodes.append((itC % 1000) / 1000)
        nodes.append((itC % 10000) / 10000)
        nodes.append(random.uniform(0, 1))
        nodes.append(random.gauss(0, 0.2))

        for i in range(MEMORY_LENGTH):
            nodes.append(self.memory[i])

        for i in range(NODE_COUNT):
            nodes.append(self.node_states[i])
            nodes.append(self.equilibrium_length_multipliers[i])
        
        for i in range(NODE_COUNT):
            summ = 0
            for j in range(len(nodes)):
                summ += nodes[j] * self.weights[k]
                k += 1
            self.node_states[i] = sigmoid(summ)
        
        for i in range(NODE_COUNT):
            summ = 0
            for j in range(len(nodes)):
                summ += nodes[j] * self.weights[k]
                k += 1
            self.equilibrium_length_multipliers[i] = 2 * sigmoid(summ)
        
        for i in range(MEMORY_LENGTH):
            summ = 0
            for j in range(len(nodes)):
                summ += nodes[j] * self.weights[k]
                k += 1
            self.memory[i] += sigmoid(summ)
            if self.memory[i] > 1:
                self.memory[i] = 1
            if self.memory[i] < -1:
                self.memory[i] = -1
        


        



class Agentfactory:
    def __init__(self):
        self.agents = []
    
    def add_random_agent(self):
        global next_id
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        node_states = []
        node_locs = []
        node_velocities = []
        for i in range(NODE_COUNT):
            dx = random.uniform(-1 * MAX_NODE_SPACE, MAX_NODE_SPACE)
            dy = random.uniform(-1 * MAX_NODE_SPACE, MAX_NODE_SPACE)
            node_states.append(random.uniform(0, 1))
            node_locs.append([x + dx, y + dy])
            node_velocities.append([random.uniform(-1 * MAX_START_VELOCITY, MAX_START_VELOCITY), random.uniform(-1 * MAX_START_VELOCITY, MAX_START_VELOCITY)])
        
        equilibrium_lengths = []
        equilibrium_length_multipliers = []
        for i in range(NODE_COUNT):
            x = node_locs[i][0]
            y = node_locs[i][1]
            for j in range(min(NODE_COUNT, i)):
                if i == j:
                    continue
                x1 = node_locs[j][0]
                y1 = node_locs[j][1]
                distance = ((x1 - x) ** 2 + (y1 - y) ** 2) ** 0.5
                equilibrium_lengths.append(distance)
                equilibrium_length_multipliers.append(1)
        
        weights = []
        for i in range(WEIGHTS_COUNT):
            weights.append(random.gauss(0, WEIGHTS_SIGMA))

        memory = []
        for i in range(MEMORY_LENGTH):
            memory.append(0)

        food = 100

        self.agents.append(Agent(next_id, node_states, node_locs, node_velocities, equilibrium_lengths, equilibrium_length_multipliers, weights, food, memory))
        next_id += 1
    
    def reproduce(self, agent):
        global next_id
        self.agents.append(copy.deepcopy(agent))
        agent.food -= 100
        self.agents[-1].id = next_id
        self.agents[-1].food = 100
        for i in range(NODE_COUNT):
            self.agents[-1].node_velocities[i][0] = 0
            self.agents[-1].node_velocities[i][1] = 0
        

        # Mutate
        random_mag = 10 ** random.randint(-6, 0)
        for i in range(WEIGHTS_COUNT):
            if random.uniform(0, 1) < 1 / WEIGHTS_COUNT:
                self.agents[-1].weights[i] += random.uniform(-1, 1) * random_mag
                if self.agents[-1].weights[i] > MAX_WEIGHT_MAG:
                    self.agents[-1].weights[i] = MAX_WEIGHT_MAG
                
                if self.agents[-1].weights[i] < -1 * MAX_WEIGHT_MAG:
                    self.agents[-1].weights[i] = -1 * MAX_WEIGHT_MAG
        
        random_mag = 10 ** random.randint(-6, 0)
        k = 0
        for i in range(NODE_COUNT):
            for j in range(min(NODE_COUNT, i)):
                if random.uniform(0, 1) < 1 / NODE_COUNT:
                    self.agents[-1].equilibrium_lengths[k] += random.uniform(-1, 1) * random_mag
                    if self.agents[-1].equilibrium_lengths[k] > MAX_NODE_SPACE:
                        self.agents[-1].equilibrium_lengths[k] = MAX_NODE_SPACE
                k += 1

        dx = random.uniform(-1 * MAX_BIRTH_DIST, MAX_BIRTH_DIST)
        dy = random.uniform(-1 * MAX_BIRTH_DIST, MAX_BIRTH_DIST)
        for i in range(NODE_COUNT):
            self.agents[-1].node_locs[i][0] += dx
            self.agents[-1].node_locs[i][1] += dy
            if self.agents[-1].node_locs[i][0] > 1:
                self.agents[-1].node_locs[i][0] = 1
            elif self.agents[-1].node_locs[i][0] < 0:
                self.agents[-1].node_locs[i][0] = 0
            
            if self.agents[-1].node_locs[i][1] > 1:
                self.agents[-1].node_locs[i][1] = 1
            elif self.agents[-1].node_locs[i][1] < 0:
                self.agents[-1].node_locs[i][1] = 0
        next_id += 1
    
    def kill_starved(self):
        for i in range(len(self.agents) - 1, -1, -1):
            if self.agents[i].food <= 0:
                self.agents.pop(i)
    
    def physics(self):
        for agent in self.agents:

            node_force_vectors = []
            for i in range(NODE_COUNT):
                node_force_vectors.append([0, 0])

            k = 0
            for i in range(NODE_COUNT):
                x = agent.node_locs[i][0]
                y = agent.node_locs[i][1]

                for j in range(min(NODE_COUNT, i)):
                    if i == j:
                        continue
                    x1 = agent.node_locs[j][0]
                    y1 = agent.node_locs[j][1]

                    distance = ((x1 - x) ** 2 + (y1 - y) ** 2) ** 0.5

                    if distance == 0 or distance > MAX_MAX_NODE_SPACE:
                        continue

                    force = (distance - agent.equilibrium_lengths[k] * agent.equilibrium_length_multipliers[k]) * FORCE_MULTIPLIER
                    k += 1

                    node_force_vectors[i][0] -= (x - x1) / distance * force
                    node_force_vectors[j][0] += (x - x1) / distance * force

                    node_force_vectors[i][1] -= (y - y1) / distance * force
                    node_force_vectors[j][1] += (y - y1) / distance * force

            for i in range(NODE_COUNT):
                agent.node_locs[i][0] += agent.node_velocities[i][0] * DELTA_T
                agent.node_locs[i][1] += agent.node_velocities[i][1] * DELTA_T

                if agent.node_locs[i][0] > 1:
                    agent.node_locs[i][0] = 1
                elif agent.node_locs[i][0] < 0:
                    agent.node_locs[i][0] = 0
                
                if agent.node_locs[i][1] > 1:
                    agent.node_locs[i][1] = 1
                elif agent.node_locs[i][1] < 0:
                    agent.node_locs[i][1] = 0

                agent.node_velocities[i][0] *= VELOCITY_DAMPENING
                agent.node_velocities[i][1] *= VELOCITY_DAMPENING

                agent.node_velocities[i][0] += node_force_vectors[i][0] * DELTA_T / (CONSTANT_MASS + agent.node_states[i])
                agent.node_velocities[i][1] += node_force_vectors[i][1] * DELTA_T / (CONSTANT_MASS + agent.node_states[i])


   
def sigmoid(x_in):
    return (1 / (1 + math.exp(-1 * x_in)))


ecosystem = Agentfactory()

for i in range(AGENT_COUNT):
    ecosystem.add_random_agent()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Close window
            running = False
        

    
    screen.fill((255, 255, 255))

    for x in range(FOOD_TILES_ACROSS):
        for y in range(FOOD_TILES_ACROSS):
            intensity = math.floor(max(min(food_grid[x][y] / 100 * 255, 255), 0))
            tile_size = math.ceil(WIDTH / FOOD_TILES_ACROSS)
            x_draw = x * tile_size
            y_draw = y * tile_size
            
            color = (intensity, intensity, intensity)
            pygame.draw.rect(screen, color, (x_draw, y_draw, tile_size, tile_size))

    for agent in ecosystem.agents:
        k = 0
        for i in range(NODE_COUNT):
            x = agent.node_locs[i][0] * WIDTH
            y = agent.node_locs[i][1] * WIDTH
            
            for j in range(min(NODE_COUNT, i)):
                if i == j:
                    continue
                x1 = agent.node_locs[j][0] * WIDTH
                y1 = agent.node_locs[j][1] * WIDTH
                intensity = math.floor(agent.equilibrium_length_multipliers[k] / 2 * 255)
                k += 1
                line_color = (0, 255 - intensity, intensity)
                pygame.draw.line(screen, line_color, (x, y), (x1, y1), LINE_WIDTH)
            
        
        for i in range(NODE_COUNT):
            x = agent.node_locs[i][0] * WIDTH
            y = agent.node_locs[i][1] * WIDTH
            intensity = math.floor(agent.node_states[i] * 255)
            color = (255 - intensity, intensity, 0)
            if i == 0:
                pygame.draw.rect(screen, (255, 255, 0), (x - round(NODE_WIDTH / 2) - 2, y - round(NODE_WIDTH / 2) - 2, NODE_WIDTH + 4, NODE_WIDTH + 4))
            pygame.draw.rect(screen, color, (x - round(NODE_WIDTH / 2), y - round(NODE_WIDTH / 2), NODE_WIDTH, NODE_WIDTH))

        agent.eat()
        if random.uniform(0, 1) < METABOLIZE_CHANCE:
            agent.metabolize()
        
        if agent.food > 200:
            ecosystem.reproduce(agent)

        agent.think()
    
    
    pygame.display.flip()

    ecosystem.kill_starved()
    ecosystem.physics()

    for i in range(FOOD_GEN_RATE):
        x = random.randint(0, FOOD_TILES_ACROSS - 1)
        y = random.randint(0, FOOD_TILES_ACROSS - 1)
        if food_grid[x][y] < 100:
            food_grid[x][y] += 1

    itC += 1

    time.sleep(1 / FRAMES_PER_SEC)


pygame.quit()

