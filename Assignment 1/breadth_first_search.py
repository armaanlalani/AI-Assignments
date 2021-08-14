from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    
    # gets the goal and initial state
    goal_states = problem.goal_states[0]
    state = problem.init_state

    # initializes the variables and data structures needed for the search
    node = Node(None,state,None,0)
    if goal_states == node.state:
        return [], 0, 0 # checks to see if the initial state is the goal state
    frontier = deque([node])
    seen = set([node.state])
    explored = set()
    max_frontier_size = 0
    num_nodes_expanded = 0
    
    while len(frontier) != 0: # continues to loop while the frontier is empty
        max_frontier_size = max(max_frontier_size, len(frontier))
        node = deque.popleft(frontier) # pops the node in the queue
        num_nodes_expanded = num_nodes_expanded + 1
        explored.add(node.state)
        seen.remove(node.state)
        actions = problem.get_actions(node.state) # obtains the actions associated with the node
        for action in actions: # iterates through the actions
            child = problem.get_child_node(node,action)
            if child.state not in explored and child.state not in seen: # checks to see if the child node has not been explored or seen before
                if child.state == goal_states:
                    return path_constructor(child,state), num_nodes_expanded, max_frontier_size # returns the path if the child is the goal state
                frontier.append(child)
                seen.add(child.state) # adds the child's attributed to the frontier and seen set

    return [], num_nodes_expanded, max_frontier_size # if the loop is broken without a path --> no solution to the problem

# function that deterines the final path once the goal state is found
def path_constructor(node, start):
    path = []
    path.insert(0,node.state) # inserts the goal state at the front of the list
    parent = node.parent
    while(True):
        path.insert(0,parent.state) # continually adds the parent node of the current node to the front of the list
        if parent.state == start: # breaks out of the loop once the initial state is reached
            break
        parent = parent.parent
    return path


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print("Final Path: " + str(path))
    print("Number of Nodes Expanded: " + str(num_nodes_expanded))
    print("Max Frontier Size: " + str(max_frontier_size) + "\n")

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print("Final Path: " + str(path))
    print("Number of Nodes Expanded: " + str(num_nodes_expanded))
    print("Max Frontier Size: " + str(max_frontier_size))