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



class VictimAgent_MC():
    
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

        # keep track of sum and count of returns for each state
        returns_sum = defaultdict(float)
        returns_count = defaultdict(float)

        policy = self.MakeEpsilonGreedyPolicy()

        for i_episode in range(num_episodes):
            episode = []
            state = self.env.reset()
            
            score = 0

            for t in range(self.Tmax):
                # choose action
                probs = policy(state)
                action = np.random.choice(np.arange(len(probs)), p=probs)
                
                # step
                next_state, reward, done, _ = self.env.step(action)
                
                # save trajectory
                episode.append((state, action, reward))
                score += reward
                
                # done?
                if done:
#                     print(f"episode = {i_episode} | timesteps = {t} | return = {score}")
                    break
                    
                state = copy.deepcopy(next_state)
                
            # get (s,a) from episode
            sa_in_episode = set([(x[0], x[1]) for x in episode])

            for state, action in sa_in_episode:
                sa_pair = (state, action)
                first_occurence_idx = next(i for i,x in enumerate(episode) if x[0]==state and x[1] == action)
                G = sum([x[2]*(self.discount_factor**i) for i,x in enumerate(episode[first_occurence_idx:])])
                returns_sum[sa_pair] += G
                returns_count[sa_pair] += 1.0
                self.Q[state][action] = returns_sum[sa_pair] / returns_count[sa_pair]

#         return Q, policy

                
                
                
if __name__ == "__main__":
    
    from envs.env3D_4x4 import GridWorld_3D_env
    
    env = GridWorld_3D_env()
    
    victim_args = {
        "env": env, 
        "discount_factor": 1.0, 
        "alpha": 0.1, 
        "epsilon": 0.1,
    }
    
    victim = VictimAgent_MC(**victim_args)
    victim.train(300)




