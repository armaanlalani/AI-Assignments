import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    diagonal_1 = np.zeros(2*N) # diagonals going from left to right --> queens along these types of diagonals share the property where their row number minus column number remain constant
    diagonal_2 = np.zeros(2*N) # diagonals going from right to left --> queens along these types of diagonals share the property where their row number plus column number remain constant
    rows = np.zeros(N) # keeps track of the number of queens in each row of the board

    for column in range(N): # update the diagonal and row arrays based on the greedy n queens initialization
        rows[initialization[column]] += 1
        diagonal_1[initialization[column] - column + N] += 1
        diagonal_2[initialization[column] + column] += 1

    for idx in range(max_steps):
        if max(rows) == 1 and max(diagonal_1) == 1 and max(diagonal_2) == 1: # if there are no conflicts in the arrays, then the algorithm is complete
            return solution, idx
        while True:
            column = np.random.randint(0,N)
            row = solution[column] # choose a random queen on the board
            if rows[row] > 1 or diagonal_1[row - column + N] > 1 or diagonal_2[row + column] > 1: # check to make sure a conflict exists --> if it does, break out of the loop
                break
        optimize(N, column, rows, diagonal_1, diagonal_2, row, solution) # calls the optimization function --> follows a similar logic to the greedy choice
        num_steps += 1
    return [], -1

def optimize(N, column, rows, diagonal_1, diagonal_2, old_row, solution): # very similar code to the greedy choice --> also updates the arrays at the end based on the new position
    min_row_numbers = [0]
    min_conflicts = rows[0] + diagonal_1[-column+N] + diagonal_2[column]
    for j in range(1,N):
        conflicts = rows[j] + diagonal_1[j-column+N] + diagonal_2[j+column]
        if conflicts < min_conflicts:
            min_row_numbers = [j]
            min_conflicts = conflicts
        elif conflicts == min_conflicts:
            min_row_numbers.append(j)
    index = np.random.randint(len(min_row_numbers))
    new_row = min_row_numbers[index]
    rows[old_row] -= 1 # subtracts 1 from the old position since the queen is not placed at another position
    diagonal_1[old_row - column + N] -= 1
    diagonal_2[old_row + column] -= 1

    solution[column] = new_row #  updates the solution to the new position chosen
    rows[new_row] += 1 # adds 1 to the new position of the arrays
    diagonal_1[new_row - column + N] += 1
    diagonal_2[new_row + column] += 1

if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 1000
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    plot_n_queens_solution(assignment_initial)

    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved)

    print(n_steps)
