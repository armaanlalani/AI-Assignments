from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

# function that implements the same functionality as breadth first search
def actions(problem, node, explored, seen, frontier, seen_other, start, end):
    explored.add(node.state)
    del seen[node.state]
    actions = problem.get_actions(node.state) # obtains the actions associated with the node
    for action in actions:
        child = problem.get_child_node(node,action)
        if child.state not in explored and child.state not in seen: # if the child node has not been explored and is not in seen
            if child.state in seen_other: # if the child node is in the other frontier (i.e. child node from source frontier is in the destination frontier)
                start_path = path_constructor(child,start) # find the path to get from the start to the common node
                end_path = path_constructor(seen_other[child.state],end) # find the path to get from the destination to the common node
                return start_path + end_path[::-1][1:] # concatenates the two paths and flips the end_path since it is backwards
            frontier.append(child)
            seen[child.state] = child
    return None

# same path_constructor function in breadth_first_search
def path_constructor(node, start):
    path = []
    path.insert(0,node.state)
    parent = node.parent
    while(True):
        path.insert(0,parent.state)
        if parent.state == start:
            break
        parent = parent.parent
    return path

def bidirectional_search(problem):
    """
        Implement a bidirectional search algorithm that takes instances of SimpleSearchProblem (or its derived
        classes) and provides a valid and optimal path from the initial state to the goal state.

        :param problem: instance of SimpleSearchProblem
        :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
                 num_nodes_expanded: number of nodes expanded by your search
                 max_frontier_size: maximum frontier size during search
        """
    
    # obtains the source and destination states
    source_state = problem.init_state
    dest_state = problem.goal_states[0]

    # instantiates all of the variables and data structures needed for the search
    node_source = Node(None,source_state,None,0)
    node_dest = Node(None,dest_state,None,0)
    if source_state == dest_state:
        return [], 0, 0 # if the source state is equal to the destination then the problem is done
    frontier_source = deque([node_source]) # frontiers for both tne source and destination nodes
    frontier_dest = deque([node_dest])
    seen_source, seen_dest = dict(), dict() # seen dictionarieis for both source and frontier
    seen_source[node_source.state] = node_source
    seen_dest[node_dest.state] = node_dest
    explored_source, explored_dest = set(), set() # explored sets for both source and frontier
    max_frontier_size = 0
    num_nodes_expanded = 0
    
    while len(frontier_source) != 0 and len(frontier_dest) != 0: # continues to loop while both frontiers are not empty
        max_frontier_size = max(len(frontier_source), len(frontier_dest), max_frontier_size)
        for i in range(0,len(frontier_source)): # pops all of the nodes in the source frontier at the beginning of the cycle
            node = deque.popleft(frontier_source)
            num_nodes_expanded = num_nodes_expanded + 1
            path = actions(problem,node,explored_source,seen_source,frontier_source,seen_dest,source_state,dest_state) # calls the action function to update variables and data structures
            if path != None:
                return path, num_nodes_expanded, max_frontier_size # if a path is found, return the path
        for i in range(0,len(frontier_dest)): # pops all of the nodes in the destination frontier at the beginning of the cycle
            node = deque.popleft(frontier_dest)
            num_nodes_expanded = num_nodes_expanded + 1
            path = actions(problem,node,explored_dest,seen_dest,frontier_dest,seen_source,dest_state,source_state) # calls the action function to update variables and data structures
            if path != None:
                return path[::-1], num_nodes_expanded, max_frontier_size # if a path is found, return a path --> backwards in this case since it is found in the destination frontier
    return [], num_nodes_expanded, max_frontier_size


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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
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
    path, num_nodes_expanded, max_frontier_size = bidirectional_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print("Final Path: " + str(path))
    print("Number of Nodes Expanded: " + str(num_nodes_expanded))
    print("Max Frontier Size: " + str(max_frontier_size))

    # Be sure to compare with breadth_first_search!