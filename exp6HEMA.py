 

import numpy as np

# Constants
GRID_SIZE = 3
NUM_ACTIONS = 4
NUM_STATES = GRID_SIZE * GRID_SIZE
START_STATE = 0
GOAL_STATE = NUM_STATES - 1
OBSTACLE_STATE = 4
EPSILON = 0.1
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
NUM_EPISODES = 1000

# Initialize Q-table
q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

# Helper function to convert 2D coordinates to 1D state index
def get_state_index(row, col):
    return row * GRID_SIZE + col

# Helper function to get possible next states from the current state
def get_next_states(state):
    row, col = divmod(state, GRID_SIZE)
    possible_next_states = []

    # Check up
    if row > 0:
        possible_next_states.append(get_state_index(row - 1, col))
    # Check down
    if row < GRID_SIZE - 1:
        possible_next_states.append(get_state_index(row + 1, col))
    # Check left
    if col > 0:
        possible_next_states.append(get_state_index(row, col - 1))
    # Check right
    if col < GRID_SIZE - 1:
        possible_next_states.append(get_state_index(row, col + 1))

    return possible_next_states

# Q-learning algorithm
for episode in range(NUM_EPISODES):
    state = START_STATE

    while state != GOAL_STATE:
        # Epsilon-greedy strategy for action selection
        if np.random.rand() < EPSILON:
            action = np.random.randint(NUM_ACTIONS)
        else:
            action = np.argmax(q_table[state, :])

        # Take the selected action and observe the next state and reward
        next_states = get_next_states(state)
        next_state = np.random.choice(next_states)
        reward = -1  # Small negative reward for each step

        # Update Q-value using the Q-learning update rule
        q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + \
                                 LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[next_state, :]))

        # Move to the next state
        state = next_state

# Print the learned Q-table
print("Learned Q-table:")
print(q_table)