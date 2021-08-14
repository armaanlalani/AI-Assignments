import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem
# import matplotlib.pyplot as plt


def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    # sets the intial goal state and initial state
    goal_state = problem.goal_states[0]
    state = problem.init_state

    # initializes all of the variables and data structures needed for the search
    node = Node(None,state,None,0)
    if goal_state == node.state:
        return [state], 0, 0
    frontier = queue.PriorityQueue() # intializes the frontier of the priority queue
    frontier.put((0,node)) # priority queue will store the path cost and the node as a tuple
    explored = {}
    explored[node.state] = 0 # initializes explored dictionary
    path = False
    max_frontier_size = 0
    num_nodes_expanded = 0
    
    while True:
        if frontier.qsize() == 0: # loops until the frontier is empty
            return [], num_nodes_expanded, max_frontier_size # if frontier is empty --> no solution to the problem
        max_frontier_size = max(max_frontier_size,frontier.qsize())
        node = (frontier.get())[1] # gets the node with the smallest cost
        if node.state == goal_state:
            break
        actions = problem.get_actions(node.state) # gets all of the actions associated with the node
        for action in actions:
            num_nodes_expanded += 1
            child = problem.get_child_node(node,action) # gets child associated with the actions
            if child.state not in explored or explored[child.state] > child.path_cost: # if the child has not been explored yet or if it has and has a higher path cost than what is already in the dictionary
                f = child.path_cost + problem.manhattan_heuristic(child.state,goal_state) # determines the path cost with the heuristic
                frontier.put((f,child))
                explored[child.state] = child.path_cost # updates the data structures accordingly

    final_path = [] # determines the final path by working backwards --> same process as breadth and bidirectional searches
    final_path.insert(0,goal_state)
    parent = node.parent
    while(True):
        final_path.insert(0,parent.state)
        if(parent.state == state):
            break
        parent = parent.parent

    return final_path, num_nodes_expanded, max_frontier_size

'''
def graphs(N):
    P = np.linspace(0.1,0.9,17)
    nodes_generated = []
    solved = []
    for p in P:
        print("Currently solving for probability " + str(p))
        A_solved = 0
        nodes = 0
        for run in range(0,100,1):
            problem = get_random_grid_problem(p,N,N)
            path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
            if path != []:
                A_solved = A_solved + 1
            nodes = nodes + num_nodes_expanded
            print("Finished run " + str(run))
        solved.append(A_solved/(run+1))
        nodes_generated.append(nodes/(run+1))
    plt.plot(P,solved)
    plt.title("Proportion of Grid Searchs Solved versus P_opp")
    plt.xlabel("P_opp")
    plt.ylabel("Proportion of Searches solved by A*")
    plt.show()
    plt.plot(P,nodes_generated)
    plt.title("Nodes Explored versus P_opp")
    plt.xlabel("P_opp")
    plt.ylabel("Average Number of Nodes Expanded")
    plt.show()
    '''

def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.27
    transition_end_probability = 0.45
    peak_nodes_expanded_probability = 0.43
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    
    p_occ = 0.3
    M = 500
    N = 500
    problem = get_random_grid_problem(p_occ, M, N)
    
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    
    # Plot the result
    problem.plot_solution(path)
    print("Number of Nodes Expanded: " + str(num_nodes_expanded))
    print("Max Frontier Size: " + str(max_frontier_size))

    # graphs(500)

    # Experiment and compare with BFS