import pygame
import random
import time
import math
import copy
import pickle
import os
import shutil


next_id = 0
AGENT_COUNT = 2000
MAX_NODE_SPACE = 0.01
MAX_MAX_NODE_SPACE = .03
NODE_COUNT = 4
DELTA_T = .03
MAX_START_VELOCITY = 0.00
VELOCITY_DAMPENING = .990
CONSTANT_MASS = 0.5
WEIGHTS_COUNT = 800
WEIGHTS_SIGMA = 0.5
MAX_WEIGHT_MAG = 3
MEMORY_LENGTH = 10
METABOLIZE_CHANCE = 0.3
MAX_BIRTH_DIST = 0.02
FORCE_MULTIPLIER = 30

SAVE_INT = 100
LOAD_FROM_SAVE = False

FOOD_TILES_ACROSS = 200
FOOD_GEN_RATE = 1600
RANDOM_AGENT_CHANCE = 0.001
MIN_AGENTS = 10
MAX_AGE = 7000

TIMELAPSE_FREQUENCY = 100

CYCLES_PER_DISPLAY = 1
FRAMES_PER_SEC = 60
NODE_WIDTH = 4
LINE_WIDTH = 2
INITIAL_FOOD = 30


# Custom Scenario
Y_MOVE_RATE = 0.018


itC = 0
time_ref = time.time()

food_grid = []
for x in range(FOOD_TILES_ACROSS):
    temp = []
    for y in range(FOOD_TILES_ACROSS):
        temp.append(INITIAL_FOOD)
    food_grid.append(temp)


pygame.init()

TIMELAPSE_FOLDER = "timelapse"
if os.path.exists(TIMELAPSE_FOLDER):
    shutil.rmtree(TIMELAPSE_FOLDER)
os.makedirs(TIMELAPSE_FOLDER)

WIDTH = 1500
HEIGHT = WIDTH
screen = pygame.display.set_mode((WIDTH, HEIGHT + 100))
font = pygame.font.Font(None, 36)

class Agent:
    def __init__(self, id, node_states, node_locs, node_velocities, equilibrium_lengths, equilibrium_length_multipliers, weights, food, memory):
        self.id = id
        self.node_states = node_states
        self.birth_node_states = copy.deepcopy(node_states)
        self.node_locs = node_locs
        self.node_velocities = node_velocities
        self.birth_node_locs = copy.deepcopy(node_locs)
        self.equilibrium_lengths = equilibrium_lengths
        self.equilibrium_length_multipliers = equilibrium_length_multipliers
        self.weights = weights
        self.food = food
        self.memory = memory
        self.age = 0

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

        x = self.node_locs[0][0]
        y = self.node_locs[0][1]

        for i in range(1, NODE_COUNT):
            dx = (self.node_locs[i][0] - x)
            dy = (self.node_locs[i][0] - y)
            nodes.append(dx / MAX_NODE_SPACE)
            nodes.append(dy / MAX_NODE_SPACE)

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
            self.equilibrium_length_multipliers[i] = 1.8 * sigmoid(summ) + .2
        
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

    def save_agents(self, filename="agents.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.agents, f)
    
    def load_agents(self, filename="agents.pkl"):
        global next_id
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        
        if isinstance(loaded, list):
            self.agents = loaded
        else:
            self.agents = [loaded]
        next_id = max(agent.id for agent in self.agents) + 1 if self.agents else 0
    
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
        
        self.agents[-1].age = 0

        # Mutate
        random_mag = 10 ** random.randint(-6, 0)
        for i in range(WEIGHTS_COUNT):
            if random.uniform(0, 1) < 1 / WEIGHTS_COUNT / 2:
                self.agents[-1].weights[i] += random.uniform(-1, 1) * random_mag
                if self.agents[-1].weights[i] > MAX_WEIGHT_MAG:
                    self.agents[-1].weights[i] = MAX_WEIGHT_MAG
                
                if self.agents[-1].weights[i] < -1 * MAX_WEIGHT_MAG:
                    self.agents[-1].weights[i] = -1 * MAX_WEIGHT_MAG
        
        random_mag = 10 ** random.randint(-6, 0)
        k = 0
        for i in range(NODE_COUNT):
            for j in range(min(NODE_COUNT, i)):
                if random.uniform(0, 1) < 1 / NODE_COUNT / 2:
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
            if (self.agents[i].food <= 0 or self.agents[i].age > MAX_AGE) and len(self.agents) > MIN_AGENTS:
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
                        k += 1
                        continue
                    x1 = agent.node_locs[j][0]
                    y1 = agent.node_locs[j][1]

                    distance = ((x1 - x) ** 2 + (y1 - y) ** 2) ** 0.5

                    if distance == 0:
                        continue

                    force = (distance - agent.equilibrium_lengths[k] * agent.equilibrium_length_multipliers[k]) * FORCE_MULTIPLIER

                    if distance > MAX_MAX_NODE_SPACE:
                        force *= 0.1
                        agent.food = -100
                    
                    k += 1

                    node_force_vectors[i][0] -= (x - x1) / distance * force
                    node_force_vectors[j][0] += (x - x1) / distance * force

                    node_force_vectors[i][1] -= (y - y1) / distance * force
                    node_force_vectors[j][1] += (y - y1) / distance * force

            for i in range(NODE_COUNT):
                agent.node_locs[i][0] += agent.node_velocities[i][0] * DELTA_T
                agent.node_locs[i][1] += agent.node_velocities[i][1] * DELTA_T + Y_MOVE_RATE * DELTA_T * (1 - agent.node_locs[i][1])

                if agent.node_locs[i][0] > 1:
                    agent.node_locs[i][0] = 1
                elif agent.node_locs[i][0] < 0:
                    agent.node_locs[i][0] = 0
                
                if agent.node_locs[i][1] > 1:
                    agent.node_locs[i][1] = 1
                elif agent.node_locs[i][1] < 0:
                    agent.node_locs[i][1] = 0

                
                agent.node_velocities[i][0] *= VELOCITY_DAMPENING * agent.node_states[i]
                agent.node_velocities[i][1] *= VELOCITY_DAMPENING * agent.node_states[i]

                agent.node_velocities[i][0] += node_force_vectors[i][0] * DELTA_T / (CONSTANT_MASS)
                agent.node_velocities[i][1] += node_force_vectors[i][1] * DELTA_T / (CONSTANT_MASS)


   
def sigmoid(x_in):
    return (1 / (1 + math.exp(-1 * x_in)))


ecosystem = Agentfactory()
if LOAD_FROM_SAVE:
    ecosystem.load_agents()
else:
    for i in range(AGENT_COUNT):
        ecosystem.add_random_agent()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Close window
            running = False
        


    if itC % CYCLES_PER_DISPLAY == 0:    
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
                    line_color = (255 - intensity, 0, intensity)
                    pygame.draw.line(screen, line_color, (x, y), (x1, y1), LINE_WIDTH)
                
            
            for i in range(NODE_COUNT):
                x = agent.node_locs[i][0] * WIDTH
                y = agent.node_locs[i][1] * WIDTH
                intensity = math.floor(agent.node_states[i] * 255)
                color = (255 - intensity, intensity, 0)
                if i == 0:
                    pygame.draw.rect(screen, (255, 255, 0), (x - round(NODE_WIDTH / 2) - 1, y - round(NODE_WIDTH / 2) - 1, NODE_WIDTH + 2, NODE_WIDTH + 2))
                pygame.draw.rect(screen, color, (x - round(NODE_WIDTH / 2), y - round(NODE_WIDTH / 2), NODE_WIDTH, NODE_WIDTH))

        total_food = sum(sum(row) for row in food_grid)
        agent_food_acc = 0
        for agent in ecosystem.agents:
            agent_food_acc += agent.food
        agent_food_acc /= len(ecosystem.agents)
        text_surface = font.render(f'Iteration: {itC}   Agents Alive: {len(ecosystem.agents)}   Amount of food on grid: {round(total_food / 1000)}   Cycles per second: {round(itC / (time.time() - time_ref))}  Average agent food: {round(agent_food_acc, 1)}', True, (255, 0, 0))
        pygame.draw.rect(screen, (100, 100, 100), (0, HEIGHT, WIDTH, 100))
        screen.blit(text_surface, (50, HEIGHT + 20))
        text_surface = font.render(f'Agents ever born: {next_id}', True, (255, 0, 0))
        screen.blit(text_surface, (50, HEIGHT + 50))
        pygame.display.flip()
        time.sleep(1 / FRAMES_PER_SEC)

    
    if itC % TIMELAPSE_FREQUENCY == 0:
        filename = os.path.join(TIMELAPSE_FOLDER, f"frame_{itC:06d}.png")
        pygame.image.save(screen, filename)


    for agent in ecosystem.agents:

        agent.eat()
        if random.uniform(0, 1) < METABOLIZE_CHANCE:
            agent.metabolize()
        
        if agent.food > 200:
            ecosystem.reproduce(agent)

        agent.think()
        agent.age += 1

    ecosystem.kill_starved()
    ecosystem.physics()

    for i in range(FOOD_GEN_RATE):
        x = random.randint(0, FOOD_TILES_ACROSS - 1)
        y = random.randint(0, FOOD_TILES_ACROSS - 1)
        if food_grid[x][y] < 100:
            food_grid[x][y] += 1
    
    if itC % SAVE_INT == 0:
        ecosystem.save_agents()
    
    if random.uniform(0, 1) < RANDOM_AGENT_CHANCE:
        ecosystem.add_random_agent()

    itC += 1

    


pygame.quit()

