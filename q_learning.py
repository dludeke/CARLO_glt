import random
import math
import numpy as np
from example_intersection import Intersection


class QLearningAgent:
    def __init__(self, env): 
        self.env = env
        self.state_space = env.observation_space
        self.state_2_idx = {s:idx for idx, s in enumerate(self.state_space)}
        self.action_space = env.action_space
        self.discount_factor = 0.95
        self.Q = np.zeros((self.state_space.size, self.action_space.size))
        self.epsilon = 0.8 # probability of random arm
        self.decay = 0.9
        self.learning_rate = 0.5

    # returns index of state in the Q matrix
    def state_2_index(self, state):
        return self.state_2_idx[state]

    def act(self, state):
        # epsilon greedy method of choosing action
        # return random action with probability epsilon
        # otherwise, return greedy action
        epsilon_greedy = random.uniform(0, 1)
        if self.epsilon > epsilon_greedy:
            self.epsilon *= self.decay
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q[state,:])
        return action

     # update the Q matrix
    def update(self, s, a, r, s_prime):
        self.Q[s,a] = self.Q[s,a] + self.learning_rate * (r + self.discount_factor * max(self.Q[s_prime,:]) - self.Q[s,a])
    
    def train(self):
        # train the model
        return 0

if __name__ == "__main__":
    # at each timestep, do the following: act, step, update reward