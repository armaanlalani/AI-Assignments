import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    greedy_init = np.zeros(N).astype(int)
    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)

    ### YOUR CODE GOES HERE
    diagonal_1 = np.zeros(2*N) # diagonals going from left to right --> queens along these types of diagonals share the property where their row number minus column number remain constant
    diagonal_1[greedy_init[0] + N] += 1 # N is added for indexing purposes
    diagonal_2 = np.zeros(2*N) # diagonals going from right to left --> queens along these types of diagonals share the property where their row number plus column number remain constant
    diagonal_2[greedy_init[0]] += 1
    rows = np.zeros(N) # keeps track of the number of queens in each row of the board
    rows[greedy_init[0]] += 1

    for i in range(1,N): # loop from the first to last column, since the 0th column was already taken care of
        min_row_numbers = [0]
        min_conflicts = rows[0] + diagonal_1[-i+N] + diagonal_2[i] # the first row we see will have minimal conflicts since it if the first one --> update the row number of number of conflicts accordingly
        for j in range(1,N):
            conflicts = rows[j] + diagonal_1[j-i+N] + diagonal_2[j+i] # determine the conflicts of the position we are in
            if conflicts < min_conflicts: # update data if the position has the least number of columns
                min_row_numbers = [j]
                min_conflicts = conflicts
            elif conflicts == min_conflicts: # add to the list if the position has the same number of conflicts as the previous minimum
                min_row_numbers.append(j)
        index = np.random.randint(len(min_row_numbers))
        greedy_init[i] = min_row_numbers[index] # pick a random index from the minimum rows
        diagonal_1[greedy_init[i] - i + N] += 1
        diagonal_2[greedy_init[i] + i] += 1
        rows[greedy_init[i]] += 1 # update the diagonal and row lists accordingly

    return greedy_init


if __name__ == '__main__':
    sol = initialize_greedy_n_queens(6)
    print(sol)