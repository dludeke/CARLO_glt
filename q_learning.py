import random
import math
import numpy as np
from Intersection import *


class QLearningAgent:
    def __init__(self, env): 
        self.env = env
        self.state_space = env.observation_space
        self.action_space = env.action_space
        self.discount_factor = 0.95
        self.Q_matrix = np.zeros(())

    def act(self, state):
        # epsilon greedy method of choosing action
        return 0


    def update(self, curstate, curaction, reward, nextstate):
        # update the Q matrix
        return 0
    
    def train(self):
        # train the model
        return 0

    if __name__ == "__main__":
        # at each timestep, do the following: act, step, update reward 
        return 0

    
    



if __name__ == "__main__":

    env = 