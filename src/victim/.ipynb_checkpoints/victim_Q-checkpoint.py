import sys
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy

from collections import namedtuple
from collections import defaultdict
from itertools import count
import itertools

import os
from os.path import dirname, abspath

import sys
if "../" not in sys.path:
    sys.path.append("../")



class VictimAgent_Q():
    
    def __init__(self, env, discount_factor=1.0, alpha=0.1, epsilon=0.1):
        self.env = env
        self.nA = self.env.action_space.n
        self.init_T = env.T.copy()
        
        self.discount_factor = discount_factor
        self.alpha = alpha
        self.epsilon = epsilon
        
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
        self.Tmax = 200
        
    def reset_Q(self):
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        
    def reset_env(self):
        self.env.reset_altitude()

    """
    Q-learning algorithm with transfer_Q
    """
    def MakeEpsilonGreedyPolicy(self):
        def policy_fn(observation):
            A = np.ones(self.nA, dtype = float) * self.epsilon/self.nA
            best_action = np.argmax(self.Q[observation])
            A[best_action] += (1.0 - self.epsilon)
            return A
        return policy_fn
    

    def train(self, num_episodes):

        # The policy we're following
        policy = self.MakeEpsilonGreedyPolicy()

        for i_episode in range(num_episodes):

            # Reset env
            state = self.env.reset()
            score = 0

            # One step in the environment
#             for t in itertools.count():
            for t in range(self.Tmax):

                # Take a step
                action_probs = policy(state)
                action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                next_state, reward, done, _ = self.env.step(action)
                score += reward

                # TD Update
                best_next_action = np.argmax(self.Q[next_state])    
                td_target = reward + self.discount_factor * self.Q[next_state][best_next_action]
                td_delta = td_target - self.Q[state][action]
                self.Q[state][action] += self.alpha * td_delta

                if done:
#                     print(f"episode = {i_episode} | timesteps = {t} | return = {score}")
                    break

                state = copy.deepcopy(next_state)
                
if __name__ == "__main__":
    
    from envs.env3D_4x4 import GridWorld_3D_env
    
    env = GridWorld_3D_env()
    
    victim_args = {
        "env": env, 
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    
    victim = VictimAgent_Q(**victim_args)
    victim.train(300)




