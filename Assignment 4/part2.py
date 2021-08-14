# part2.py: Project 4 Part 2 script
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
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (4ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    # np.random.seed(1) # TODO: Remove this

    policy = np.random.randint(len(env.actions), size=(len(env.states), 1))
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    unchanged = False # unchanged variable as specified in algorithm
    for i in range(max_iter): # loops through the number of iterations
      policy_evaluation(env,agent,policy,max_iter) # calls policy evaluation defined below
      unchanged = True
      for state in env.states: # iterates through each of the states
        utility = np.zeros(len(env.actions)) 
        for state_next in env.states:
          for action in env.actions: # iterates through the states with each respective action
            utility[action] += env.transition_model[state,state_next,action]*agent.utility[state_next,0] # finds the utility of each respective action
        if policy[state] != np.argmax(utility):
          policy[state] = np.argmax(utility) # updates the policy if a new maximum is found
          unchanged = False
      if unchanged: # breaks if policy was not updated
        break
    return np.reshape(policy,(len(env.states)))

def policy_evaluation(env, agent, policy, max_iter, eps = 0.0001): # given a policy --> find utility if policy is executed
  for i in range(max_iter):
    delta = 0
    for state in env.states: # iterates through the states
      new = env.rewards[state]
      for state_next in env.states: # iterates through the states --> moving from first to second state
        new += env.transition_model[state,state_next,policy[state]]*agent.utility[state_next]*agent.gamma # new utility value of respective state
      delta = max(delta,abs(agent.utility[state]-new)) # updates the value of delta
      agent.utility[state] = new
    if delta <= eps: # breaks if the change is less than the max change allowed
      break
    ## END: Student code