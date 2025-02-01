"""
This script creates a pickle file containing a single “special” agent.
The agent is configured to start near the left side of the screen,
has high energy, and is preset with a slight rightward velocity so that it
“moves across the screen.” The agent’s neural parameters (weights, node states,
and so on) are chosen to favor strong outputs.
"""

import pickle
import math
import random
import copy

# --------------- Parameters (match these to your simulation) ---------------
NODE_COUNT = 5           # How many nodes the agent has
MAX_NODE_SPACE = 0.02    # Maximum offset when arranging nodes
WEIGHTS_COUNT = 500      # Total number of weights expected in the agent
MEMORY_LENGTH = 3        # Number of memory values the agent carries
FOOD_INITIAL = 1000      # Give the agent plenty of food (energy)
SPECIAL_AGENT_ID = 0     # The ID for this special agent

# --------------- Helper Function ---------------
def sigmoid(x_in):
    return 1 / (1 + math.exp(-x_in))

# --------------- Agent Class Definition ---------------
# (This must match the definition your simulation expects.)
class Agent:
    def __init__(self, id, node_states, node_locs, node_velocities,
                 equilibrium_lengths, equilibrium_length_multipliers,
                 weights, food, memory):
        self.id = id
        # Save a copy of the initial node states/locations for reference.
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

    def think(self):
        # (The simulation’s think method will be called; here we leave it unchanged.)
        k = 0
        nodes = []
        nodes.append(1)
        # In your simulation these time-based inputs are provided by a global itC.
        # Here we just use 0 for simplicity.
        nodes.append(0)
        nodes.append(0)
        nodes.append(0)
        nodes.append(0)
        nodes.append(0)
        nodes.append(random.uniform(0, 1))
        nodes.append(random.gauss(0, 0.2))

        for i in range(MEMORY_LENGTH):
            nodes.append(self.memory[i])
        for i in range(NODE_COUNT):
            nodes.append(self.node_states[i])
            nodes.append(self.equilibrium_length_multipliers[i])
        
        # Update node_states (a very simple “brain”)
        for i in range(NODE_COUNT):
            summ = 0
            for j in range(len(nodes)):
                summ += nodes[j] * self.weights[k]
                k += 1
            self.node_states[i] = sigmoid(summ)
        
        # Update equilibrium multipliers
        for i in range(NODE_COUNT):
            summ = 0
            for j in range(len(nodes)):
                summ += nodes[j] * self.weights[k]
                k += 1
            self.equilibrium_length_multipliers[i] = 1.8 * sigmoid(summ) + .2
        
        # Update memory
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

# --------------- Create the Special Agent ---------------
# We want this agent to be “destined to do especially well” by giving it:
#   - A starting location on the left side (e.g. x=0.1, y=0.5 in normalized coordinates).
#   - A slight positive (rightward) velocity on its head (node 0).
#   - High energy ("food") so it does not starve.
#   - Node states preset to 1 (so they are near the maximum) and weights set high
#     (to drive strong outputs when its brain “thinks”).

# Pre-configure the node states: set all to 1 (maximizing the damping factor in physics).
node_states = [1.0 for _ in range(NODE_COUNT)]

# Arrange node locations in a horizontal line so that node 0 is the "head."
# (Coordinates are normalized to [0,1].)
node_locs = []
base_x = 0.1  # Start near the left edge.
base_y = 0.5  # Center vertically.
for i in range(NODE_COUNT):
    # Space nodes out by MAX_NODE_SPACE so the agent has a formation.
    node_locs.append([base_x + i * MAX_NODE_SPACE, base_y])

# Set node velocities:
# Give node 0 a small rightward velocity; other nodes start at rest.
node_velocities = []
for i in range(NODE_COUNT):
    if i == 0:
        node_velocities.append([0.005, 0.0])  # rightward velocity
    else:
        node_velocities.append([0.0, 0.0])

# Calculate equilibrium lengths and multipliers between nodes.
equilibrium_lengths = []
equilibrium_length_multipliers = []
for i in range(NODE_COUNT):
    x_i, y_i = node_locs[i]
    for j in range(i):
        x_j, y_j = node_locs[j]
        dist = math.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
        equilibrium_lengths.append(dist)
        equilibrium_length_multipliers.append(1)

# Set the weights for the neural “brain.”
# Here we use a constant high value (e.g. 10.0) so that every summation is large,
# driving the sigmoid function nearly to 1.
weights = [10.0 for _ in range(WEIGHTS_COUNT)]

# Set the agent's food (energy) to a high level.
food = FOOD_INITIAL

# Initialize memory to all zeros.
memory = [0.0 for _ in range(MEMORY_LENGTH)]

# Create the agent.
special_agent = Agent(SPECIAL_AGENT_ID,
                      node_states,
                      node_locs,
                      node_velocities,
                      equilibrium_lengths,
                      equilibrium_length_multipliers,
                      weights,
                      food,
                      memory)

# --------------- Save the Agent to a Pickle File ---------------
with open("special_agent.pkl", "wb") as f:
    pickle.dump(special_agent, f)

print("Created special_agent.pkl with one preconfigured agent!")
