import random
import math
import numpy as np
from example_intersection import *


class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_values = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        if (state, action) not in self.q_values:
            self.q_values[(state, action)] = 0.0
        return self.q_values[(state, action)]

    def compute_value_from_q_values(self, state):
        q_values = [self.get_q_value(state, action) for action in self.actions]
        if len(q_values) == 0:
            return 0.0
        return max(q_values)

    def compute_action_from_q_values(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.get_q_value(state, action) for action in self.actions]
        if len(q_values) == 0:
            return None
        max_q_value = max(q_values)
        if q_values.count(max_q_value) > 1:
            best_actions = [i for i in range(len(self.actions)) if q_values[i] == max_q_value]
            return self.actions[random.choice(best_actions)]
        else:
            return self.actions[q_values.index(max_q_value)]

    def get_action(self, state):
        return self.compute_action_from_q_values(state)

    def update(self, state, action, next_state, reward):
        sample = reward + self.gamma * self.compute_value_from_q_values(next_state)
        self.q_values[(state, action)] = (1 - self.alpha) * self.get_q_value(state, action) + self.alpha * sample
