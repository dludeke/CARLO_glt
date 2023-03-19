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
    



if __name__ == "__main__":

    env = 