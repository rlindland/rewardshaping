"""
Iterated Prisoner's dilemma environment.
"""
import gym
import numpy as np
import itertools 
import matplotlib 
import matplotlib.style 
from matplotlib import pyplot as plt
import pandas as pd 
import sys 
  
  
from collections import defaultdict 
import plotting 
  
matplotlib.style.use('ggplot') 

from gym.spaces import Discrete, Tuple

from common import OneHot

#pip install gym==0.10.3

class IteratedPrisonersDilemma(gym.Env):
    """
    A two-agent vectorized environment for the Prisoner's Dilemma game.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    NAME = 'IPD'
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, max_steps):
        self.max_steps = max_steps
        ###
        ###   C        D   
        ### C -1, -1   -3, 0
        ### D 0 -3     -2, -2 
        self.payout_mat = np.array([[3., 0.], [4., 1.]])
        self.action_space = \
            Tuple([Discrete(self.NUM_ACTIONS), Discrete(self.NUM_ACTIONS)])
        self.observation_space = \
            Tuple([OneHot(self.NUM_STATES), OneHot(self.NUM_STATES)])

        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.NUM_STATES)
        init_state[-1] = 1
        observations = [self.NUM_STATES-1, self.NUM_STATES-1]
        return observations

    def step(self, action):
        ac0, ac1 = action
        self.step_count += 1
        ### 0 is cooperate, 1 is defect
        rewards = [self.payout_mat[ac0][ac1], self.payout_mat[ac1][ac0]]

        #state = np.zeros(self.NUM_STATES)
        state = ac0*2 + ac1
        #state[ac0 * 2 + ac1] = 1
        observations = [state, state]

        done = (self.step_count == self.max_steps)
        return observations, rewards, done

env = IteratedPrisonersDilemma(100000)

def createEpsilonGreedyPolicy(Q, epsilon, num_actions): 
    """ 
    Creates an epsilon-greedy policy based 
    on a given Q-function and epsilon. 
       
    Returns a function that takes the state 
    as an input and returns the probabilities 
    for each action in the form of a numpy array  
    of length of the action space(set of possible actions). 
    """
    def policyFunction(state): 
   
        Action_probabilities = np.ones(num_actions, 
                dtype = float) * epsilon / num_actions 
        
        best_action = np.argmax(Q[state]) 
        Action_probabilities[best_action] += (1.0 - epsilon) 
        return Action_probabilities 
   
    return policyFunction 

def qLearning(env, num_episodes, discount_factor = 0.9, 
                            alpha1 = 0.01, alpha2 = 0.02, epsilon = 0.1): 
    """ 
    Q-Learning algorithm: Off-policy TD control. 
    Finds the optimal greedy policy while improving 
    following an epsilon-greedy policy"""
       
    # Action value function 
    # A nested dictionary that maps 
    # state -> (action -> action-value). 

    #Q+ shaping
    #Q1 = {}
    #Q1[]

    Q1 = {}
    for i in range(5):
        Q1[i]=[0.,0.]
    Q1[0]=[198./19., 220./19.]
    Q1[1]=[144./19., 160./19.]
    Q1[2]=[144./19., 160./19.]
    Q1[3]=[198./19., 220./19.]
    Q1[4]=[0.,0.]

    #Q2=defaultdict(lambda: np.zeros(2))
    Q2 = {}
    for i in range(5):
        Q2[i]=[0.,0.]

    # Keeps track of useful statistics 
    stats1 = []
    stats2 = []  
       
    # Create an epsilon greedy policy function 
    # appropriately for environment action space 
    policy1 = createEpsilonGreedyPolicy(Q1, epsilon, 2)
    policy2 = createEpsilonGreedyPolicy(Q2, epsilon, 2) 
       
    # For every episode 
    for ith_episode in range(1): 
           
        # Reset the environment and pick the first action 
        state = env.reset()[0]
           
        for t in itertools.count(): 
               
            # get probabilities of all actions from current state 
            action_probabilities1 = policy1(state)
            action_probabilities2 = policy2(state) 
   
            # choose action according to  
            # the probability distribution 
            action1 = np.random.choice(np.arange( 
                      len(action_probabilities1)), 
                       p = action_probabilities1)

            action2 = np.random.choice(np.arange( 
                      len(action_probabilities2)), 
                       p = action_probabilities2) 
   
            # take action and get reward, transit to next state 
            next_state, reward, done = env.step([action1, action2]) 
            next_state = next_state[0]
            # Update statistics 
            stats1.append(reward[0])
            stats2.append(reward[1])
               
            # TD Update 
            best_next_action = np.argmax(Q1[next_state])     
            td_target = reward[0] + discount_factor * Q1[next_state][best_next_action] 
            td_delta = td_target - Q1[state][action1] 
            Q1[state][action1] += alpha1 * td_delta

            best_next_action = np.argmax(Q2[next_state])     
            td_target = reward[1] + discount_factor * Q2[next_state][best_next_action] 
            td_delta = td_target - Q2[state][action2] 
            Q2[state][action2] += alpha2 * td_delta  
            # done is True if episode terminated    
            if done: 
                break
                   
            state = next_state

    stats1 = np.array(stats1)
    stats2 = np.array(stats2)

    stats1 = np.convolve(stats1,np.ones(5000,dtype=float),'valid') / 5000
    stats2 = np.convolve(stats2,np.ones(5000,dtype=float),'valid') / 5000
       
    return Q1, Q2, stats1, stats2 



Q1, Q2, stats1, stats2 = qLearning(env, 10) 
#print(Q1)
#print(Q2)
#print(stats1)
#print(stats2)
np.save("Q1.npy", Q1)
np.save("Q2.npy", Q2)
np.save("stats1.npy", stats1)
np.save("stats2.npy", stats2)

plt.figure(1)
plt.plot(stats1[:100000], 'ro', markersize=3)
plt.plot(stats2[:100000], 'bo', markersize=3)
plt.show()

