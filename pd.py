"""
Iterated Prisoner's dilemma environment.
"""
import gym
import numpy as np
import itertools 
import matplotlib 
import matplotlib.style 
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
        self.payout_mat = np.array([[-1., -3.], [0., -2.]])
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

env = IteratedPrisonersDilemma(10000)

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

def qLearning(env, num_episodes, discount_factor = 1.0, 
                            alpha = 0.6, epsilon = 0.1): 
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

    Q1 = defaultdict(lambda: np.zeros(2)) 
    Q2 = defaultdict(lambda: np.zeros(2)) 
   
    # Keeps track of useful statistics 
    stats1 = plotting.EpisodeStats( 
        episode_lengths = np.zeros(num_episodes), 
        episode_rewards = np.zeros(num_episodes))
    stats2 = plotting.EpisodeStats( 
        episode_lengths = np.zeros(num_episodes), 
        episode_rewards = np.zeros(num_episodes))     
       
    # Create an epsilon greedy policy function 
    # appropriately for environment action space 
    policy1 = createEpsilonGreedyPolicy(Q1, epsilon, 2)
    policy2 = createEpsilonGreedyPolicy(Q2, epsilon, 2) 
       
    # For every episode 
    for ith_episode in range(num_episodes): 
           
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
            stats1.episode_rewards[ith_episode] += reward[0]
            stats1.episode_lengths[ith_episode] = t 

            stats2.episode_rewards[ith_episode] += reward[1]
            stats2.episode_lengths[ith_episode] = t 
               
            # TD Update 
            best_next_action = np.argmax(Q1[next_state])     
            td_target = reward[0] + discount_factor * Q1[next_state][best_next_action] 
            td_delta = td_target - Q1[state][action1] 
            Q1[state][action1] += alpha * td_delta

            best_next_action = np.argmax(Q2[next_state])     
            td_target = reward[1] + discount_factor * Q2[next_state][best_next_action] 
            td_delta = td_target - Q2[state][action2] 
            Q2[state][action2] += alpha * td_delta  
            # done is True if episode terminated    
            if done: 
                break
                   
            state = next_state 
       
    return Q1, Q2, stats1, stats2 



Q1, Q2, stats1, stats2 = qLearning(env, 100) 
print(Q1)
print(Q2)
print(stats1)
print(stats2)