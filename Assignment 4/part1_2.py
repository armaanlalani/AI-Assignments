# part1_2.py: Project 4 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311 Winter 2020
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Dr. Jonathan Kelly
# jkelly@utias.utoronto.ca
#
# Teaching Assistant:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca

###
# Imports
###

import numpy as np
from mdp_env import mdp_env
from mdp_agent import mdp_agent

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the value_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
    """
    value_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 653). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs
    ---------------
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        eps:   Max error allowed in the utility of a state
        max_iter: Max iterations for the algorithm

    Outputs
    ---------------
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    policy = np.empty_like(env.states)
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    for i in range(max_iter): # loop through all of the iterations
      delta = 0
      for state in env.states:
        utilities = np.zeros(len(env.actions)) # holds the utilities for each action which will be updated later
        for state_next in env.states:
          for action in env.actions:
            utilities[action] += env.transition_model[state,state_next,action]*agent.utility[state_next,0] # finds the utility of each respective action
        new = env.rewards[state] + np.max(utilities)*agent.gamma # new utility value of the respective state
        delta = max(delta,abs(new-agent.utility[state,0])) # updates the delta value accordingly
        agent.utility[state,0] = new # updates the new utility value in the utility array
        policy[state] = np.argmax(utilities) # action that maximizes the utility
      if delta < eps*(1-agent.gamma)/agent.gamma: # breaks out of the loop if the change between the utility values is below the threshold
        break
    ## END Student code
    return policy