import numpy as np
# Define the grid world
# 'S' represents the starting point
# 'G' represents the goal
# 'H' represents a hazard/obstacle
# '_' represents empty cells
grid_world = np.array([
 ['S', '_', '_', 'H'],
 ['_', 'H', '_', 'H'],
 ['_', '_', '_', 'H'],
 ['H', '_', '_', 'G']
])
# Define the rewards matrix
# -1 for moving into an obstacle/hazard
# 0 for all other transitions
# 100 for reaching the goal
rewards = np.where(grid_world == 'G', 100, 0)
rewards[grid_world == 'H'] = -1
# Define the Q-table
num_states = np.prod(grid_world.shape)
num_actions = 4 # up, down, left, right
Q = np.zeros((num_states, num_actions))
# Define the helper functions
def get_possible_actions(state):
    row, col = np.argwhere(np.ravel(grid_world == 'S'))[0]
    possible_actions = []
    if row > 0 and grid_world[row - 1, col] != 'H':
        possible_actions.append(0) # Up
    if row < grid_world.shape[0] - 1 and grid_world[row + 1, col] != 'H':
        possible_actions.append(1) # Down
    if col > 0 and grid_world[row, col - 1] != 'H':
        possible_actions.append(2) # Left
    if col < grid_world.shape[1] - 1 and grid_world[row, col + 1] != 'H':
        possible_actions.append(3) # Right
    return possible_actions

def get_next_state(state, action):
     row, col = np.unravel_index(state, grid_world.shape)
     if action == 0: # Up
        row -= 1
     elif action == 1: # Down
        row += 1
     elif action == 2: # Left
        col -= 1
     elif action == 3: # Right
        col += 1
# Q-learning parameters
alpha = 0.1 # Learning rate
gamma = 0.9 # Discount factor
epsilon = 0.1 # Epsilon-greedy policy parameter
# Q-learning algorithm
num_episodes = 1000
for episode in range(num_episodes):
 state = np.argwhere(np.ravel(grid_world == 'S'))[0][0]
 done = False
 while not done:
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice(get_possible_actions(state))
    else:
        action = np.argmax(Q[state])
    next_state = get_next_state(state, action)
    reward = rewards[np.unravel_index(next_state, grid_world.shape)]
    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
    if grid_world[np.unravel_index(next_state, grid_world.shape)] == 'G':
        done = True
    else:
        state = next_state
# Find the optimal path
path = ['S']
current_state = np.argwhere(np.ravel(grid_world == 'S'))[0][0]
while grid_world[np.unravel_index(current_state, grid_world.shape)] != 'G':
    action = np.argmax(Q[current_state])
    next_state = get_next_state(current_state, action)
    path.append(grid_world[np.unravel_index(next_state, grid_world.shape)])
    current_state = next_state
print("Optimal Path:", path)