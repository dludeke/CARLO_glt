import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time
from qlearning import QLearningAgent


# Initialize the environment and agents
env = IntersectionEnv()
c1 = Car(Point(60, 0), Point(0, 0), Point(0, 0), 0)
c2 = Pedestrian(Point(60, 120), Point(0, 0))
q_agent = QLearningAgent()

# Run the simulation
for i in range(100):
    # Update the environment with the current positions of the agents
    env.update(c1, c2)

    # Get the state of c1
    state = env.get_state(c1)

    # Use q-learning to select an action for c1
    action = q_agent.select_action(state)

    # Update c1's velocity and heading based on the selected action
    c1.update(action)

    # Move c1 and c2
    c1.move()
    c2.move()

    # Check for collisions
    if env.check_collision(c1, c2):
        print("Collision detected!")
        break

    # Update the Q-table based on the current state, action, and rewards
    reward = env.get_reward(c1, c2)
    next_state = env.get_state(c1)
    q_agent.update(state, action, reward, next_state)

# Print the final Q-table
print(q_agent.q_table)
