import random
import math
import numpy as np
from example_intersection import Intersection


class QLearningAgent:
    def __init__(self, env): 
        self.env = env
        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.discount_factor = 0.95
        self.Q_matrix = np.zeros((self.state_space.size, self.action_space.size))
        self.epsilon = 0.8 # probability of random arm
        self.decay = 0.9

    def act(self, state):
        # epsilon greedy method of choosing action
        # return random action with probability epsilon
        # otherwise, return greedy action
        epsilon_greedy = random.uniform(0, 1)
        if self.epsilon > epsilon_greedy:
            self.epsilon *= self.decay
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Qmat[state,:])
        return action



    def update(self, curstate, curaction, reward, nextstate):
        # update the Q matrix
        return 0
    
    def train(self):
        # train the model
        return 0

if __name__ == "__main__":
    # at each timestep, do the following: act, step, update reward 
    return 0

    
    
