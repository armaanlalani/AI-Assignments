
# part3_solution.py  (adopted from the work of Anson Wong)
#
# --
# Artificial Intelligence
# ROB 311 Winter 2021
# Programming Project 4
#
# --
# University of Toronto Institute for Aerospace Studies
# Stars Lab
#
# Course Instructor:
# Matthew Giamou
# mathhew.giamau@robotics.utias.utoronto.ca
#
# Teaching Assistant:
# Sepehr Samavi
# sepehr@robotics.utias.utoronto.ca
#
# Abhinav Grover
# abhinav.grover@robotics.utias.utoronto.ca

"""
 We set up bandit arms with fixed probability distribution of success,
 and receive stochastic rewards from each arm of +1 for success,
 and 0 reward for failure.
"""
import numpy as np

class MAB_agent:
    """
        TODO:
        Implement the get_action and update_state function of an agent such that it 
        is able to maximize the reward on the Multi-Armed Bandit (MAB) environment.
    """
    def __init__(self, num_arms=5):
        np.random.seed(10)
        self.greedy = True
        if self.greedy:
            self.__num_arms = num_arms #private
            self.success = np.zeros((num_arms,3)) # array where dimension 2 holds reward, dimension 3 holds times machine has run, dimension 1 holds the mean payout
            self.epsilon = 0.5 # parameter for probability of choosing a random machine
            self.current_max = 0 # stores the current maximum
            self.current_max_percent = 0 # stores the maximum payout of the current maximum
            self.turns = 0
        else:
            self.__num_arms = num_arms
            self.turns = 0
            self.success = np.zeros((num_arms,3)) + 1e-5
            self.a = np.ones((num_arms,1))

    def update_state(self, action, reward):
        """
            TODO:
            Based on your choice of algorithm, use the the current action and 
            reward to update the state of the agent. 
            Optinal function, only use if needed.
        """
        if self.greedy:
            self.success[action,2] += 1
            if reward == 1:
                self.success[action,1] += 1
            self.success[action,0] = self.success[action,1]/self.success[action,2] # updates the success array according to its definition
            self.current_max = np.argmax(self.success[:,0]) # finds the machine with the highest payout
            self.current_max_percent = self.success[self.current_max,0] # gets the highest payout value
        else:
            self.success[action,0] += reward
            self.success[action,1] += 1
            self.success[action,2] = self.success[action,0]/self.success[action,1]
            self.a = 2*np.log(self.turns/self.success[:,1])

    def get_action(self) -> int:
        """
            TODO:
            Based on your choice of algorithm, generate the next action based on
            the current state of your agent.
            Return the index of the arm picked by the policy.
        """
        if self.greedy:
            move = np.random.choice([0,1],p=[self.epsilon,1-self.epsilon]) # determines a random move vs choosing the max based on epsilon
            if move == 0 and self.current_max_percent < 0.75: # if the move chosen is to be random and the current maximum payout is less than 0.75
                choice = np.random.randint(0,self.__num_arms) # then choose a random move
                if self.success[choice,0] < 0.5 and self.success[choice,2] != 0: # only proceed with the random move if its payout is above 0.5
                    return self.current_max
                return choice
            self.turns += 1
            if self.turns == 50 or self.turns == 100:
                self.epsilon *= 0.6
            if self.turns == 200:
                self.epsilon = 0.05
            return self.current_max # otherwise choose the machine with highest payout
        else:
            if self.turns < self.__num_arms:
                move = self.turns
            else:
                maximum = self.success[:,2] + self.a
                moves = np.argwhere(maximum == np.amax(maximum)).flatten().tolist()
                move = moves[np.random.randint(0,len(moves))]
                move = np.argmax(maximum)
                # print(move)
            self.turns += 1
            return move
