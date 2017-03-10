"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import os
from random import randint
import logging
import sys
from isolation import Board
from copy import deepcopy
from copy import copy

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


# vasiliy.mikhailov: for reviewer - please do not review transposition table, it is under development. Idea is to collect
# winning combinations and apply deep neural network from sk-learn.
# I will not be able to complete until March, 15, probably will complete later

# vasiliy.mikhailov: transposition table for game agent, where key is binary representation (to lookup values faster)
# and value is the scores for that table
# NB: binary representation of board is 112-bit number, first bit is always on, second is 0 if player 1 turn,1 if player 2
# turn, then goes X coord of player 1 (3 bits), Y coord of player 1 (3 bits), X coord of player 2 (3 bits), Y coord of
# player 2 (3 bits). Then goes 1 if player is maximizing. Next 98 are (width of board * height of board 2 bits) go in pairs.
# Each pair of bits represent 00 if cell is not occupied by players, 01 - occupied by maximizing player, 02 - occupied
# by minimizing player. Leftmost pair of bits is for top-left corner of board, then board is iterated
# [i ascending, j ascending] first on j, then on i

# vasiliy.mikhailov: depth of transposition table, zero means "turn off transposition table activities"
TRANSPOSITION_TABLE_DEPTH = 0

# vasiliy.mikhailov: number of transposition table hits (for debug)
transposition_table_hits = [0 for i in range(49)]

# vasiliy.mikhailov: number of transposition table increments (for debug)
transposition_table_increments = [0 for i in range(49)]

# vasiliy.mikhailov: file name for transposition table
# data will come in form of "board binary representation<space>next move x<space>next move y>move count from game start<space>score\n"
TRANSPOSITION_TABLE_FILENAME = './transposition_table.csv'

# vasiliy.mikhailov: load transposition table. Data will be shared between all CustomPlayers
transposition_table = {}
if os.path.isfile(TRANSPOSITION_TABLE_FILENAME):
    f = open(TRANSPOSITION_TABLE_FILENAME, 'r+')
    for line in f:
        (binary_board, score, move_x, move_y, move_count) = line.split()
        transposition_table[int(binary_board)] = (float(score), int(move_x), int(move_y), int(move_count))

    f.close()

# vasiliy.mikhailov: number of timeouts occured for learning rate study
variables = {'timeouts' : 0}

# vasiliy.mikhailov: const for logger
logger = logging.getLogger('game_agent')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# vasiliy.mikhailov: global variables to tune up score functions, can be set outside module
player_weights = {"player": 0.3125, "opponent": 1.0}
in_isolation = {'number_of_moves': 30}

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    return chase_in_isolation_score(game, player)

def chase_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. Chase score by default goes with weight of player legal moves of 0.3125 and weight of
    opponent legal moves of 1.0, which gives 83% wins over Improved score in standard tournament of 150 ms per
    move

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        score = float("-inf")
    elif game.is_winner(player):
        score = float("inf")
    else:
        opponent = game.get_opponent(player)

        player_legal_moves = game.get_legal_moves(player)
        opponent_legal_moves = game.get_legal_moves(opponent)

        score = player_weights["player"] * len(player_legal_moves) - player_weights["opponent"] * len(opponent_legal_moves)
    return score

def independent_moves(game, legal_moves, number_of_turns):
    """ vasiliy.mikhailov: Helper function for next_move score. Retuns list of moves for player if he could make
    independently one move, then another move etc.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    legal_moves : list
        A list of legal moves, pass here player's initial position
    number_of_turns : Int
        Number of turns for player to move
    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    while number_of_turns > 0:
        # vasiliy.mikhailov: counting number of legal moves, it will help us to exit earlier in algorythm if no improvement can be made
        num_legal_moves_before = len(legal_moves)

        # vasiliy.mikhailov: calculating new list of legal moves by adding new moves
        directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        moves_next_turn = [(r + dr, c + dc) for (r, c) in legal_moves for dr, dc in directions if game.move_is_legal((r + dr, c + dc))]
        legal_moves = list(set(legal_moves).union(set(moves_next_turn)))

        # vasiliy.mikhailov: checking if new moves were added, if none - quit cycle
        num_legal_moves_after = len(legal_moves)
        if num_legal_moves_before == num_legal_moves_after:
            break

        number_of_turns -= 1

    return legal_moves

def next_move_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. Works as Improved score if number of legal moves of player
    and number of legal moves of opponent differ, if same - takes into account number of
    moves that can be made after making current moves

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        score = float("-inf")
    elif game.is_winner(player):
        score = float("inf")
    else:
        opponent = game.get_opponent(player)

        player_legal_moves = game.get_legal_moves(player)
        opponent_legal_moves = game.get_legal_moves(opponent)

        if len(player_legal_moves) == len(opponent_legal_moves):
            player_extended_moves = independent_moves(game, player_legal_moves, 1)
            opponent_extended_moves = independent_moves(game, opponent_legal_moves, 1)

            return 0.01 * len(player_extended_moves) - 0.01 * len(opponent_extended_moves)

        score = len(player_legal_moves) - len(opponent_legal_moves)

    return score

def in_isolation_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. Returns Improved score until move 30, after that
    checks if players are isolated and if so - returns open move score
    for player

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        score = float("-inf")
    elif game.is_winner(player):
        score = float("inf")
    else:
        opponent = game.get_opponent(player)

        player_legal_moves = game.get_legal_moves(player)
        opponent_legal_moves = game.get_legal_moves(opponent)

        # vasiliy.mikhailov: after move N check if players are isolated and if they are - return open move score
        if game.move_count > in_isolation['number_of_moves']:
            all_player_moves = independent_moves(game, player_legal_moves, 49)
            all_opponent_moves = independent_moves(game, opponent_legal_moves, 49)

            intersection = list(set(all_player_moves).intersection(set(all_opponent_moves)))

            if not intersection:
                score = len(player_legal_moves)

                return score

        score = len(player_legal_moves) - len(opponent_legal_moves)
    return score

def chase_in_isolation_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player. Returns Chase score until move 30, after that
    checks if players are isolated and if so - returns open move score
    for player

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        score = float("-inf")
    elif game.is_winner(player):
        score = float("inf")
    else:
        opponent = game.get_opponent(player)

        player_legal_moves = game.get_legal_moves(player)
        opponent_legal_moves = game.get_legal_moves(opponent)

        # vasiliy.mikhailov: after move N check if players are isolated and if they are - return open move score
        if game.move_count > in_isolation['number_of_moves']:
            all_player_moves = independent_moves(game, player_legal_moves, 49)
            all_opponent_moves = independent_moves(game, opponent_legal_moves, 49)

            intersection = list(set(all_player_moves).intersection(set(all_opponent_moves)))

            if not intersection:
                score = len(player_legal_moves)

                return score

        score = player_weights["player"] * len(player_legal_moves) - player_weights["opponent"] * len(opponent_legal_moves)
    return score

# vasiliy.mikhailov: bunch of methods below is for rotating the board. We will rotate board 90, 180, 270 degrees,
# flip it vertically, horizontally and along diagonal axes 1 and 2 to store all possible combinations in our transposition table
# For algorythm to be easier to read, all actions are as following: move coordinate system to center of board (-3, -3),
# perform action, move coordinate system back (+3, +3)

def _identity_transformation(coordinate):
    """
    # vasiliy.mikhailov: performs no transformation, returns same input, required later for algorythms to be uniform
    :param x: x-coord of original cell in board
    :param y: y-coord of original cell in board
    :return:

    new_x, new_y - coordinates in resulting board
    """

    x, y = coordinate[0], coordinate[1]

    # vasiliy.mikhailov: moving coordinate system to center x, y => X, Y, doing nothing with X, Y and moving
    # coordinate system back X, Y => x, y
    # -3 and + 3 left for easier understanding of algorythm

    x, y = (x - 3) + 3, (y - 3) + 3

    return (x, y)

def _horizontal_flip(coordinate):
    """
    # vasiliy.mikhailov: performs flip over horizontal axix
    :param x: x-coord of original cell in board
    :param y: y-coord of original cell in board
    :return:

    new_x, new_y - coordinates in resulting board
    """

    x, y = coordinate[0], coordinate[1]

    # vasiliy.mikhailov: moving coordinate system to center x, y => X, Y, mirroring over X axis via multiplying X by -1
    # and moving coordinate system back X, Y => x, y
    x, y = (x - 3) * (-1) + 3, (y - 3) + 3

    return (x, y)

def _vertical_flip(coordinate):
    """
    # vasiliy.mikhailov: performs flip over vertical axix
    :param x: x-coord of original cell in board
    :param y: y-coord of original cell in board
    :return:

    new_x, new_y - coordinates in resulting board
    """

    x, y = coordinate[0], coordinate[1]

    # vasiliy.mikhailov: moving coordinate system to center x, y => X, Y, mirroring over Y axis via multiplying Y by -1
    # and moving coordinate system back X, Y => x, y

    x, y = (x - 3) + 3, (y - 3) * (-1) + 3

    return (x, y)

def _rot_180(coordinate):
    """
    # vasiliy.mikhailov: performs 180 degrees rotation
    :param x: x-coord of original cell in board
    :param y: y-coord of original cell in board
    :return:

    new_x, new_y - coordinates in resulting board
    """

    x, y = coordinate[0], coordinate[1]

    # vasiliy.mikhailov: moving coordinate system to center x, y => X, Y, mirroring over X axis via multiplying X by -1
    # and mirroring over Y axis via multiplying Y by -1 and moving coordinate system back X, Y => x, y

    x, y = (x - 3) * (-1) + 3, (y - 3) * (-1) + 3

    return (x, y)


def _flip_diag_1(coordinate):
    """
    # vasiliy.mikhailov: performs flip over first diagonal
    :param x: x-coord of original cell in board
    :param y: y-coord of original cell in board
    :return:

    new_x, new_y - coordinates in resulting board
    """

    x, y = coordinate[0], coordinate[1]

    # vasiliy.mikhailov: moving coordinate system to center x, y => X, Y, swapping X and Y
    # and moving coordinate system back X, Y => x, y

    x, y = (y - 3) + 3, (x - 3) + 3

    return (x, y)

def _flip_diag_2(coordinate):
    """
    # vasiliy.mikhailov: performs flip over second diagonal
    :param x: x-coord of original cell in board
    :param y: y-coord of original cell in board
    :return:

    new_x, new_y - coordinates in resulting board
    """

    x, y = coordinate[0], coordinate[1]

    # vasiliy.mikhailov: moving coordinate system to center x, y => X, Y, mirroring over X via multiplying by -1
    # mirroring over Y via multiplying by -1, swapping X and Y and moving coordinate system back X, Y => x, y
    x, y = (y - 3) * (-1) + 3, (x - 3) * (-1) + 3

    return (x, y)

def _rot_90(coordinate):
    """
    # vasiliy.mikhailov: performs 90 degrees flip of board
    :param x: x-coord of original cell in board
    :param y: y-coord of original cell in board
    :return:

    new_x, new_y - coordinates in resulting board
    """

    x, y = coordinate[0], coordinate[1]

    # vasiliy.mikhailov: moving coordinate system to center x, y => X, Y, swapping X and Y,
    # multiplying first coordinate by -1 and moving coordinate system back X, Y => x, y

    x, y = (y - 3) * (-1) + 3, (x - 3) + 3

    return (x, y)

def _rot_270(coordinate):
    """
    # vasiliy.mikhailov: performs 270 degrees flip of board (or minus 90 which is the same)
    :param x: x-coord of original cell in board
    :param y: y-coord of original cell in board
    :return:

    new_x, new_y - coordinates in resulting board
    """

    x, y = coordinate[0], coordinate[1]

    # vasiliy.mikhailov: moving coordinate system to center x, y => X, Y, swapping X and Y,
    # multiplying second coordinate by -1 and moving coordinate system back X, Y => x, y
    x, y = (y - 3) + 3, (x - 3) * (-1) + 3

    return (x, y)

def _board_bits(board, player_1_last_move, player_2_last_move, player, maximizing_player):
    """
    # vasiliy.mikhailov: calculate hash for board
    :param
        board: board in form of list of lists
        player_1_last_move (int, int) - player 1 last move, put here None if he did not make move
        player_2_last_move (int, int) - player_2_last_move last move, put here None if he did not make move
        player - player who wants to make next move
        maximizing_player - True for player, False for his opponent
    :return:
        transformed_board_bits     - compact replesentation of board as bit number as described above
    """

    # vasiliy.mikhailov: storing binary 1 in left part of binary representation
    transformed_board_bits = 1

    # append player 1 position in form of x * 8 + y. If he did not move put 7, 7 instead of position
    transformed_board_bits = transformed_board_bits * 8 * 8
    if player_1_last_move:
        transformed_board_bits = transformed_board_bits + player_1_last_move[0] * 8 + player_1_last_move[1]
    else:
        transformed_board_bits = transformed_board_bits + 7 * 8 + 7

    # append opponent last move in form of x * 8 + y
    transformed_board_bits = transformed_board_bits * 8 * 8
    if player_2_last_move:
        transformed_board_bits = transformed_board_bits + player_2_last_move[0] * 8 + player_2_last_move[1]
    else:
        transformed_board_bits = transformed_board_bits + 7 * 8 + 7

    # vasiliy.mikhailov: append player
    transformed_board_bits = transformed_board_bits * 2
    if player == 1:
        transformed_board_bits = transformed_board_bits + 0
    else:
        transformed_board_bits = transformed_board_bits + 1

    # vasiliy.mikhailov: append if player is maximizing
    transformed_board_bits = transformed_board_bits * 2
    if maximizing_player:
        transformed_board_bits = transformed_board_bits + 1
    else:
        transformed_board_bits = transformed_board_bits + 0

    for i in range(0, 7):
        for j in range(0, 7):
            # vasiliy.mikhailov shift 2 bits left
            transformed_board_bits = transformed_board_bits * 4

            # vasiliy.mikhailov fill left two bits with proper player signature
            transformed_board_bits = transformed_board_bits + board[i][j]

    return transformed_board_bits

def _board_variants(board, player_1_last_move, player_2_last_move, player, maximizing_player):
    """
    # vasiliy.mikhailov: calculate 8 possible transformations for board:
    (rotate 0, 90, 180, 270, flip horizonaly, flip verticaly, flip across diagonal 1 and flip across diagonal 2)
    and return them
    :param
        board: board in form of list of lists
        player_1_last_move (int, int) - player 1 last move, put here None if he did not make move
        player_2_last_move (int, int) - player_2_last_move last move, put here None if he did not make move
        player - player who wants to make next move
        maximizing_player - True for player, False for his opponent

    :return:
        list of board variants: [(transformed_board_bits, transform_function, reverse_transform_function)]
        transformed_board_bits     - compact representation of board as bit number as described above
        transform_function         - function to transform movement to coordinates of transformed board
        reverse_transform_function - function to transform movement (x, y) back into coordinates of original board

    """

    # vasiliy.mikhailov: list of forward and reverse transformations
    # forward transformation <=> reverse transformation
    # rotate 0               <=> rotate 0
    # rotate 90              <=> rotate 270
    # rotate 180             <=> rotate 180
    # rotate 270             <=> rotate 90
    # flip horizontal        <=> flip horizontal
    # flip_vertical          <=> flip vertical
    # flip_diagonal 1        <=> flip_diagonal 1
    # flip_diagonal 2        <=> flip_diagonal 2

    # vasiliy.mikhailov: ... and in code it looks like this
    transformations = {
        _identity_transformation: _identity_transformation,
        _rot_90                 : _rot_270,
        _rot_180                : _rot_180,
        _rot_270                : _rot_90,
        _horizontal_flip        : _horizontal_flip,
        _vertical_flip          : _vertical_flip,
        _flip_diag_1            : _flip_diag_1,
        _flip_diag_2            : _flip_diag_2
    }

    # vasiliy.mikhailov: placeholder for 8 transformations
    result = []

    # vasiliy.mikhailov: iterate over list of transformations, create empty board, fill it with transformed variant of
    # original board, compute binary representation and return binary representations along with reverse transform
    # function
    for (transform_function, reverse_transform_function) in transformations.items():
        transformed_board = [[0 for i in range(7)] for j in range(7)]

        for i in range(0, 7):
            for j in range(0, 7):
                (new_i, new_j) = transform_function((i, j))
                transformed_board[new_i][new_j] = board[i][j]

        transformed_player_1_last_move = transform_function(player_1_last_move) if player_1_last_move else None
        transformed_player_2_last_move = transform_function(player_2_last_move) if player_2_last_move else None

        transformed_board_bits = _board_bits(transformed_board, transformed_player_1_last_move, transformed_player_2_last_move, player, maximizing_player)

        # vasiliy.mikhailov: append transformed board along with transformation to result set
        result.append((transformed_board_bits, transform_function, reverse_transform_function))

    # vasiliy.mikhailov: return list of transformed boards and reverse transformations
    return result

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # vasiliy.mikhailov: start game from random position
        #if game.move_count == 0:
            #return (3, 3)

        # vasiliy.mikhailov: if there is no legal moves - exit immediately with illegal move
        if not legal_moves:
            return (-1, -1)

        # vasiliy.mikhailov: variable that will store our best move, initialized with first "random" move and will be replaced with better move later
        best_move = legal_moves[randint(0, len(legal_moves) - 1)]

        # vasiliy.mikhailov: variable that will store best score
        best_score = float("-inf")

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            # vasiliy.mikhailov: creating list of depth iterations. For iterative search put number of empty cells on board, for fixed search put just one iteration with depth
            iteration_list = list(range(1, game.width * game.height + 1 - game.move_count)) if self.iterative else [self.search_depth]

            for depth in iteration_list:
                score, move = self.minimax(game, depth) if self.method == 'minimax' else self.alphabeta(game, depth)

                # vasiliy.mikhailov: if better score found - save it
                if score > best_score:
                    best_score = score
                    best_move = move

                # vasiliy.mikhailov: if found +inf (means winning situation) or -inf (resign) - break cycle to save time
                if best_score in [float("+inf"), float("-inf")]:
                        break

        # vasiliy.mikhailov: this exception block is for adding values to opening book
        except Timeout:
            # vasiliy.mikhailov: do nothing, we already have best (possible) move in best_move variable that will be returned later
            variables['timeouts'] = variables['timeouts'] + 1
        except:
            # vasiliy.mikhailov: some other exception like out of memory - re-raise it
            raise
        else:
            pass

        # vasiliy.mikhailov: print combination
        #logger.debug("Move #{0} player {1} to ({2}, {3}) score: {4}".format(game.move_count, game.__player_symbols__[self], best_move[0], best_move[1], best_score))

        # Return the best move from the last completed search iteration
        return best_move
        
    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        
        # vasiliy.mikhailov: stop processing in case of no time left for move
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # vasiliy.mikhailov: for depth 0 calculate and return score of maximizing player
        if depth == 0:
            best_score = self.score(game, self)
            best_move = (-1, -1)

            # vasiliy.mikhailov: return best score and best move
            return (best_score, best_move)
        else:
            # vasiliy.mikhailov: calculating legal moves for active player
            legal_moves = game.get_legal_moves()
                
            # vasiliy.mikhailov: for maximizing player and minimizing player will be different algorythms
            if maximizing_player:
                
                # vasiliy.mikhailov: placeholders for better score and move (if any)
                best_score = float("-inf")
                best_move = (-1, -1)
            
                # vasiliy.mikhailov: traversing all legal moves
                for m in legal_moves:
                    
                    # vasiliy.mikhailov: making copy of board, making move, getting forecasted board and calculating minimax recursively for that board
                    score_candidate = self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)[0]
                    
                    # vasiliy.mikhailov: if score is better than we have, then store that best score and move for later
                    if score_candidate > best_score:
                        best_score = score_candidate
                        best_move = m

                        if best_score == float("+inf"):
                            # vasiliy.mikhailov: save score into transposition table
                            self.save_score(game, maximizing_player, best_score, best_move)
                # vasiliy.mikhailov: return best score and best move
                return (best_score, best_move)
            # vasiliy.mikhailov: for minimizing player: looking for min value instead of max and updating beta instead of alpha
            else:
                # vasiliy.mikhailov: placeholders for better score and move (if any) 
                best_score = float("+inf")
                best_move = (-1, -1)
                
                # vasiliy.mikhailov: traversing all legal moves
                for m in legal_moves:
                    # vasiliy.mikhailov: making copy of board, making move, getting forecasted board and calculating minimax recursively for that board
                    score_candidate = self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)[0]
                    
                    # vasiliy.mikhailov: if score is better than we have, then store that best score and move for later
                    if score_candidate < best_score:
                        best_score = score_candidate
                        best_move = m

                        if best_score == float("-inf"):
                            # vasiliy.mikhailov: save score into transposition table
                            self.save_score(game, maximizing_player, best_score, best_move)
                # vasiliy.mikhailov: return best score and best move
                return (best_score, best_move)

    def lookup_score(self, game, maximizing_player):
        # vasiliy.mikhailov: looking for recipe in transposition table
        if game.move_count < TRANSPOSITION_TABLE_DEPTH:
            # vasiliy.mikhailov: computing board in binary representation

            transformed_board_bits = _board_bits(
                game.__board_state__,
                game.get_player_location(game.__player_1__),
                game.get_player_location(game.__player_2__),
                game.__player_symbols__[self],
                maximizing_player
            )

            # vasiliy.mikhailov: looking up variant in transposition table, if found - return it
            # and return "+inf" for score and transformed move
            if transformed_board_bits in transposition_table:
                info = transposition_table[transformed_board_bits]
                best_score = info[0]
                best_move = (info[1], info[2])
                move_count = info[3]
                transposition_table_hits[move_count] += 1

                # # vasiliy.mikhailov: if no timeout, no other exception
                # if best_score == float("+inf"):
                #     logger.debug(
                #         "Transposition table hit: player {0} wins on move {1}".format(game.__player_symbols__[self], game.move_count))
                #     logger.debug("{0} {1} {2}".format(game.to_string(), self.score(game, self), best_move))

                if not game.move_is_legal(best_move):
                    logger.warning("Found illegal move in transposition table {0} for game {1}".format(best_move, game.to_string()))

                return (best_score, best_move)

        return (None, None)

    def save_score(self, game, maximizing_player, best_score, best_move):
        if game.move_count < TRANSPOSITION_TABLE_DEPTH:
            # vasiliy.mikhailov: computing 8 board variants and saving them to disk
            board_variants = _board_variants(game.__board_state__,
                                             game.get_player_location(game.__player_1__),
                                             game.get_player_location(game.__player_2__),
                                             game.__player_symbols__[self],
                                             maximizing_player
                                             )

            # vasiliy.mikhailov: save best move for this board to memory and disk
            f = open(TRANSPOSITION_TABLE_FILENAME, "a+")
            for (binary_board, transform_function, reverse_transform_function) in board_variants:
                if binary_board not in transposition_table:
                    transformed_move = transform_function(best_move)
                    transposition_table[binary_board] = (
                        best_score, transformed_move[0], transformed_move[1], game.move_count)
                    print(binary_board, best_score, transformed_move[0], transformed_move[1],
                          game.move_count, file=f)
                    transposition_table_increments[game.move_count] += 1

            f.close()

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        cache : dictionary
            Stores heuristic and move evaluated earlier
        level : int
            Iteration number

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """

        # vasiliy.mikhailov: stop processing in case of no time left for move
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # vasiliy.mikhailov: calculating legal moves for active player
        legal_moves = game.get_legal_moves()

        # vasiliy.mikhailov: use transposition table only with custom score function to see progress during tournament
        if self.score is custom_score:
            best_score, best_move = self.lookup_score(game, maximizing_player)

            if best_score and best_move:
                legal_moves = [best_move]

        # vasiliy.mikhailov: for depth 0 calculate and return score of maximizing player
        if depth == 0:
            best_score = self.score(game, self)
            best_move = (-1, -1)

            # vasiliy.mikhailov: return best score and best move
            return (best_score, best_move)
        else:
            # vasiliy.mikhailov: for maximizing player and minimizing player will be different algorythms
            if maximizing_player:
                # vasiliy.mikhailov: placeholders for better score and move (if any)
                best_score = float("-inf")
                best_move = (-1, -1)
            
                # vasiliy.mikhailov: traversing all legal moves
                for m in legal_moves:
                    # vasiliy.mikhailov: making copy of board, making move, getting forecasted board and calculating alphabeta recursively for that board
                    score_candidate = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, not maximizing_player)[0]
                    
                    # vasiliy.mikhailov: if score is better than we have, then store that best score and move for later
                    if score_candidate > best_score:
                        best_score = score_candidate
                        best_move = m

                    if best_score == float("+inf"):
                        # vasiliy.mikhailov: save score into transposition table
                        self.save_score(game, maximizing_player, best_score, best_move)

                    # vasiliy.mikhailov: updating alpha with best score if best score is better than alpha
                    alpha = max(alpha, best_score)

                    # vasiliy.mikhailov: alpha-beta pruning addition to minimax algorythm - stop processing if no better value could be achieved
                    if beta <= alpha:
                        break

                # vasiliy.mikhailov: return best score and best move
                return (best_score, best_move)

            # vasiliy.mikhailov: for minimizing player: looking for min value instead of max and updating beta instead of alpha
            else:
                # vasiliy.mikhailov: placeholders for better score and move (if any) 
                best_score = float("+inf")
                best_move = (-1, -1)
                
                # vasiliy.mikhailov: traversing all legal moves
                for m in legal_moves:
                    # vasiliy.mikhailov: making copy of board, making move, getting forecasted board and calculating alphabeta recursively for that board
                    score_candidate = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, not maximizing_player)[0]
                    
                    # vasiliy.mikhailov: if score is better than we have, then store that best score and move for later
                    if score_candidate < best_score:
                        best_score = score_candidate
                        best_move = m

                        if best_score == float("-inf"):
                            # vasiliy.mikhailov: save score into transposition table
                            self.save_score(game, maximizing_player, best_score, best_move)
                        
                    # vasiliy.mikhailov: updating beta with best score if best score is better than beta
                    beta = min(beta, best_score)
                    
                    # vasiliy.mikhailov: alpha-beta pruning addition to minimax algorythm - stop processing if no better value could be achieved
                    if beta <= alpha:
                        break

                # vasiliy.mikhailov: return best score and best move
                return (best_score, best_move)
