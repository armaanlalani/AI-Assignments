from abc import ABC, abstractmethod
import numpy as np


class SingleMoveGamePlayer(ABC):
    """
    Abstract base class for a symmetric, zero-sum single move game player.
    """
    def __init__(self, game_matrix: np.ndarray):
        self.game_matrix = game_matrix
        self.n_moves = game_matrix.shape[0]
        super().__init__()

    @abstractmethod
    def make_move(self) -> int:
        pass


class IteratedGamePlayer(SingleMoveGamePlayer):
    """
    Abstract base class for a player of an iterated symmetric, zero-sum single move game.
    """
    def __init__(self, game_matrix: np.ndarray):
        super(IteratedGamePlayer, self).__init__(game_matrix)

    @abstractmethod
    def make_move(self) -> int:
        pass

    @abstractmethod
    def update_results(self, my_move, other_move):
        """
        This method is called after each round is played
        :param my_move: the move this agent played in the round that just finished
        :param other_move:
        :return:
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class UniformPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(UniformPlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """

        :return:
        """
        return np.random.randint(0, self.n_moves)

    def update_results(self, my_move, other_move):
        """
        The UniformPlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class FirstMovePlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(FirstMovePlayer, self).__init__(game_matrix)

    def make_move(self) -> int:
        """
        Always chooses the first move
        :return:
        """
        return 0

    def update_results(self, my_move, other_move):
        """
        The FirstMovePlayer player does not use prior rounds' results during iterated games.
        :param my_move:
        :param other_move:
        :return:
        """
        pass

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        pass


class CopycatPlayer(IteratedGamePlayer):
    def __init__(self, game_matrix: np.ndarray):
        super(CopycatPlayer, self).__init__(game_matrix)
        self.last_move = np.random.randint(self.n_moves)

    def make_move(self) -> int:
        """
        Always copies the last move played
        :return:
        """
        return self.last_move

    def update_results(self, my_move, other_move):
        """
        The CopyCat player simply remembers the opponent's last move.
        :param my_move:
        :param other_move:
        :return:
        """
        self.last_move = other_move

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.)
        :return:
        """
        self.last_move = np.random.randint(self.n_moves)


def play_game(player1, player2, game_matrix: np.ndarray, N: int = 1000) -> (int, int):
    """

    :param player1: instance of an IteratedGamePlayer subclass for player 1
    :param player2: instance of an IteratedGamePlayer subclass for player 2
    :param game_matrix: square payoff matrix
    :param N: number of rounds of the game to be played
    :return: tuple containing player1's score and player2's score
    """
    p1_score = 0.0
    p2_score = 0.0
    n_moves = game_matrix.shape[0]
    legal_moves = set(range(n_moves))
    for idx in range(N):
        move1 = player1.make_move()
        move2 = player2.make_move()
        if move1 not in legal_moves:
            print("WARNING: Player1 made an illegal move: {:}".format(move1))
            if move2 not in legal_moves:
                print("WARNING: Player2 made an illegal move: {:}".format(move2))
            else:
                p2_score += np.max(game_matrix)
                p1_score -= np.max(game_matrix)
            continue
        elif move2 not in legal_moves:
            print("WARNING: Player2 made an illegal move: {:}".format(move2))
            p1_score += np.max(game_matrix)
            p2_score -= np.max(game_matrix)
            continue
        player1.update_results(move1, move2)
        player2.update_results(move2, move1)

        p1_score += game_matrix[move1, move2]
        p2_score += game_matrix[move2, move1]
    return p1_score, p2_score


class StudentAgent(IteratedGamePlayer):
    """
    My algorithm makes use of probabilities as well as randomized choices to attempt to choose the optimal move.
    The self.count array is essentially a 3x3x3 matrix where the the first and second dimension encode the previous moves made by both my agent and the opposition.
    The third dimension encodes the move the opponent makes following the previous move. At each move, the agent will use the counts associated with the previous
    move to develop the probabilities associated with the opponent's following move.
    Using this probability distribution, my agent will then select the move to counter the opposition's move to the best of its ability. The self.previous_move
    array holds both my agent's and the opposition's previous move which is then used to update the table as the game progresses. The move that beats the opponent's
    likeliest move is then calculated and returned.
    In order to introduce some randomness to my agent, a randomzied move is chosen 1% of the time in order to make the agent somewhat unpredictable. The random
    move is only chosen if the probability distribution of the previous move does not favour one particular move. For example, my player's agent will always
    pick a move with 100% probability distribution regardless of if a random move is selected based on the 1% probability.
    """
    def __init__(self, game_matrix: np.ndarray):
        """
        Initialize your game playing agent. here
        :param game_matrix: square payoff matrix for the game being played.
        """
        super(StudentAgent, self).__init__(game_matrix)
        self.count = np.zeros((3,3,3))
        self.moves = 0
        self.previous_move = [0,0]
        self.random = 0.01

    def make_move(self) -> int:
        """
        Play your move based on previous moves or whatever reasoning you want.
        :return: an int in (0, ..., n_moves-1) representing your move
        """
        move_type = np.random.choice([0,1],1,p=[1-self.random,self.random])
        if self.moves == 0:
            move = np.random.randint(0, self.n_moves)
            self.moves += 1
            return move
        else:
            self.moves += 1
            probability = self.count[self.previous_move[0],self.previous_move[1],:]
            if np.sum(probability) == 0:
                move = np.random.randint(0,self.n_moves)
                return move
            probability = probability / np.sum(probability)
            if move_type[0] == 1 and np.count_nonzero(probability) > 1:
                move = np.random.randint(0, self.n_moves)
                return move
            move = np.random.choice([0,1,2],1,p=probability)[0]
            move = (move + self.n_moves + 1) % self.n_moves
            return move
        

    def update_results(self, my_move, other_move):
        """
        Update your agent based on the round that was just played.
        :param my_move:
        :param other_move:
        :return: nothing
        """
        if self.moves == 1:
            self.previous_move = [my_move,other_move]
            pass
        else:
            self.count[self.previous_move[0],self.previous_move[1],other_move] += 1
            self.previous_move = [my_move,other_move]

    def reset(self):
        """
        This method is called in between opponents (forget memory, etc.).
        :return: nothing
        """
        self.count = np.zeros((3,3,3))
        self.moves = 0
        self.previous_move = [0,0]


if __name__ == '__main__':
    """
    Simple test on standard rock-paper-scissors
    The game matrix's row (first index) is indexed by player 1 (P1)'s move (i.e., your move)
    The game matrix's column (second index) is indexed by player 2 (P2)'s move (i.e., the opponent's move)
    Thus, for example, game_matrix[0, 1] represents the score for P1 when P1 plays rock and P2 plays paper: -1.0 
    because rock loses to paper.
    """
    game_matrix = np.array([[0.0, -1.0, 1.0],
                            [1.0, 0.0, -1.0],
                            [-1.0, 1.0, 0.0]])
    uniform_player = UniformPlayer(game_matrix)
    first_move_player = FirstMovePlayer(game_matrix)
    uniform_score, first_move_score = play_game(uniform_player, first_move_player, game_matrix)

    print("Uniform player's score: {:}".format(uniform_score))
    print("First-move player's score: {:}".format(first_move_score))

    # Now try your agent
    student_player = StudentAgent(game_matrix)
    copycat_player = CopycatPlayer(game_matrix)
    student_score, first_move_score = play_game(student_player, copycat_player, game_matrix)

    print("Your player's score: {:}".format(student_score))
    print("Opponent's score: {:}".format(first_move_score))