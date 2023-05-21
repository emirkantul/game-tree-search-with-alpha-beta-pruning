"""
Emir Kantul 27041

1. The Game's Characteristics:
    - The game board comprises NxN slants, which corresponds to a slightly larger (N+1)x(N+1) grid of intersections.
    - The game is Zero-Sum, meaning that the combined score of both players will always be zero. This is because each successful move by a player results in that player gaining points while the opponent loses an equivalent amount.

2. Slant Representations:
    - Each slant on the board can be in three possible states: empty, filled by player 1, or filled by player 2.
    - In the code, we use:
        * 0 to represent an empty slant (' '),
        * 1 to represent a slant filled by player 1 ('⟋'), and 
        * 2 to represent a slant filled by player 2 ('⟍').

3. Board Representations:
    - The board is made up of intersections that are either empty or have a number.
    - '*' is used to denote an empty intersection,
    - Any other number represents the value assigned to a filled intersection.

4. Players:
    - There are two players in this game. Player 1 represents the human participant, and Player 2 represents the AI opponent.

5. Game States:
    - The state of the game at any moment is defined by the status of the board's intersections, the slants on the board, the current turn, and the scores of the players.

6. Initial State:
    - When the game begins, the board is empty with no slants, it's player 1's (human player's) turn, and the scores for both players are set to 0.

7. Terminal State:
    - The game reaches a terminal state when all possible moves have been made, i.e., all slants are placed. At this point, the game is over and the scores are evaluated.

8. State Transition Function:
    - This function governs the rules of the game. Given the current state, it returns a new state where the next player has placed a slant.
    - If the placement of the slant completes an intersection, the player who placed the slant receives a score increase equivalent to the number on the intersection, while the opposing player's score decreases by the same amount.

9. Payoff Function:
    - Once the game reaches its terminal state, the Payoff function calculates the total points of each player to determine the winner.
"""
import random
import sys
import time

from tabulate import tabulate

vs_file = "test0.txt"
init_board = []
MAX_DEPTH = 3  # Maximum depth for the minimax algorithm (if None it means god mode)


class SlantState:
    def __init__(self, board, slants, turn, scores):
        self.board, self.slants, self.turn, self.scores = board, slants, turn, scores

    def int_to_str_representation(self, i, conversion):
        if conversion == "slants":
            return "/" if i == 1 else "\\" if i == 2 else " "
        elif conversion == "board":
            return "*" if i == -1 else str(i)

    def __str__(self):
        # Generate a string representation of the state for display
        temp = ""
        for i in range(len(self.slants) * 2 + 1):
            for j in range(len(self.slants) * 2 + 1):
                temp += (
                    self.int_to_str_representation(init_board[i // 2][j // 2], "board")
                    if i % 2 == 0 and j % 2 == 0
                    else "-"
                    if i % 2 == 0
                    else "|"
                    if j % 2 == 0
                    else self.int_to_str_representation(
                        self.slants[i // 2][j // 2], "slants"
                    )
                )
            temp += "\n"
        return temp


is_terminal = lambda state: 0 not in (
    num for sublist in state.slants for num in sublist
)  # Check if a state is terminal
payoff = (
    lambda state: state.scores[0] - state.scores[1]
)  # Calculate the payoff for a terminal state
possible_moves = lambda state: [
    (i, j)
    for i in range(len(state.slants))
    for j in range(len(state.slants))
    if state.slants[i][j] == 0
]  # Generate possible moves for a state


# Make a move on the current state and return the updated state
def make_move(state, move):
    i, j = move
    new_slants = [row.copy() for row in state.slants]
    new_slants[i][j] = state.turn
    new_scores, new_board = update_scores_board(state, i, j, new_slants)
    return SlantState(new_board, new_slants, 3 - state.turn, new_scores)


# Update scores and board after a move is made
def update_scores_board(state, i, j, new_slants):
    new_scores = state.scores.copy()
    new_board = [row.copy() for row in state.board]
    map_turn = {1: [(i + 1, j), (i, j + 1)], 2: [(i, j), (i + 1, j + 1)]}
    map_score = {1: [0, 1], 2: [1, 0]}
    for idx in map_turn[state.turn]:
        if new_board[idx[0]][idx[1]] != -1:
            new_board[idx[0]][idx[1]] -= 1
            if new_board[idx[0]][idx[1]] == 0:
                new_scores[map_score[state.turn][0]] += init_board[idx[0]][idx[1]]
                new_scores[map_score[state.turn][1]] -= init_board[idx[0]][idx[1]]
                new_board[idx[0]][idx[1]] = -1
    return new_scores, new_board


# Implementation of the minimax algorithm with alpha-beta pruning
def minimax(state, depth, alpha, beta, maximizing_player):
    if depth == 0 or is_terminal(state):
        return payoff(state), None
    if maximizing_player:
        maxEval = float('-inf')
        best_move = None
        for move in possible_moves(state):
            eval, _ = minimax(make_move(state, move), depth - 1, alpha, beta, False)
            if eval > maxEval:
                maxEval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return maxEval, best_move
    else:
        minEval = float('inf')
        best_move = None
        for move in possible_moves(state):
            eval, _ = minimax(make_move(state, move), depth - 1, alpha, beta, True)
            if eval < minEval:
                minEval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return minEval, best_move


# Get the best move for AI based on the minimax algorithm
def get_ai_move(state, depth, maximize):
    if maximize:
        _, move = minimax(
            state,
            (MAX_DEPTH if MAX_DEPTH is not None else depth),
            float('-inf'),
            float('inf'),
            True,
        )
    else:
        _, move = minimax(
            state,
            (MAX_DEPTH if MAX_DEPTH is not None else depth),
            float('-inf'),
            float('inf'),
            False,
        )

    return move


# Read the board state from a file
def read_board(file_path):
    with open(file_path, "r") as file:
        return [[int(i) if i != "*" else -1 for i in line.strip()] for line in file]


# Display a spinning animation
def spinner():
    chars = "|/-\\"
    for char in chars:
        sys.stdout.write('\r' + 'Running...' + char)
        time.sleep(0.1)
        sys.stdout.flush()


# Execute the game logic for AI vs AI mode or Human vs AI mode
def game_logic(current_state, record_moves, file=None):
    grid_size = len(current_state.board) - 1
    while not is_terminal(current_state):
        spinner()
        if record_moves:
            print(current_state, file=file)
            print("Player", current_state.turn, "to move.", file=file)
        if current_state.turn == 1:
            move = get_ai_move(current_state, grid_size, False)
        else:
            move = get_ai_move(current_state, grid_size, True)
        current_state = make_move(current_state, move)
        if record_moves:
            print("Player", current_state.turn, "moved to", move, file=file)
            print(file=file)
    return current_state


def init_state(board):
    if (len(board) - 1) % 2 != 0:
        raise Exception("Board size must be odd, and grid size must be even.")
    return SlantState(
        board,
        [[0 for _ in range(len(board) - 1)] for _ in range(len(board) - 1)],
        1,
        [0, 0],
    )


# Run the game for different test files and collect metrics
def game_metrics(record_moves=True):
    metrics, test_files = [], [f"test{i}.txt" for i in range(6)]
    move_files = [f"ai-vs-ai-moves/moves{i}.txt" for i, _ in enumerate(test_files)]
    for i, test_file in enumerate(test_files):
        sys.stdout.write('\nRunning Test File: ' + test_file + '\n')
        board = read_board(f"boards/{test_file}")
        global init_board
        init_board = board
        initial_state = init_state(board)

        start_time = time.time()
        if record_moves:
            with open(move_files[i], "w") as file:
                final_state = game_logic(initial_state, record_moves, file)
        else:
            final_state = game_logic(initial_state, record_moves)
        end_time = time.time()
        board_size = len(board)
        grid_size = len(board) - 1
        metrics.append(
            {
                "Test File": test_file,
                "Board Size": board_size,
                "Grid Size": grid_size,
                "Possible States": 2 ** (grid_size * grid_size),
                "Max. Possible Search Depth": grid_size * grid_size,
                "Current Search Depth": MAX_DEPTH
                if MAX_DEPTH is not None
                else grid_size * grid_size,
                "Time Taken": f"{end_time - start_time:.2f} seconds",
                "Final Scores": final_state.scores,
            }
        )
    print()
    print(tabulate(metrics, headers="keys", tablefmt="pretty"))


# Function to get a human move (Only for Human vs Ai)
def get_human_move(state):
    # Get the list of possible moves
    possible = possible_moves(state)

    # Ask for input
    while True:
        move = input("Enter your move in the format 'x y': ")
        move = tuple(map(int, move.split()))

        if move in possible:
            return move
        else:
            print("Invalid move. Please enter a valid move.")


# Run the game AI versus Human mode and record the moves
def human_vs_ai():
    metrics, record = [], f"human-vs-ai-moves/moves-{vs_file}"
    sys.stdout.write('\nRunning Test File: ' + vs_file + '\n')
    board = read_board(f"boards/{vs_file}")
    global init_board
    init_board = board
    initial_state = init_state(board)

    with open(record, "w") as file:
        current_state = initial_state
        grid_size = len(current_state.board) - 1
        while not is_terminal(current_state):
            print(current_state)
            print("Player", current_state.turn, "to move.")
            print(current_state, file=file)
            print("Player", current_state.turn, "to move.", file=file)
            if current_state.turn == 1:
                move = get_human_move(current_state)
            else:
                move = get_ai_move(current_state, grid_size, True)
            current_state = make_move(current_state, move)
            print("Player", current_state.turn, "moved to", move)
            print("Player", current_state.turn, "moved to", move, file=file)
            print()
        print(current_state)
        print("Game Over. Final scores: ", current_state.scores)
        print(current_state, file=file)
        print("Game Over. Final scores: ", current_state.scores, file=file)
        board_size = len(board)
        grid_size = len(board) - 1
        metrics.append(
            {
                "Test File": vs_file,
                "Board Size": board_size,
                "Grid Size": grid_size,
                "Possible States": 2 ** (grid_size * grid_size),
                "Max. Possible Search Depth": grid_size * grid_size,
                "Current Search Depth": MAX_DEPTH
                if MAX_DEPTH is not None
                else grid_size * grid_size,
                "Final Scores": current_state.scores,
            }
        )
        print()
        print(tabulate(metrics, headers="keys", tablefmt="pretty"))
        print(file=file)
        print(tabulate(metrics, headers="keys", tablefmt="pretty"), file=file)


# Start the game and prompt the user for the game mode
def start_game():
    choice = None
    while choice not in ['1', '2', '3', '4']:
        choice = input(
            "Choose game mode:\n1. AI vs AI with moves written to .txt\n2. AI vs AI without moves written to .txt\n3. Human vs AI\n4. Exit\nYour choice: "
        )
        if choice == '1':
            game_metrics()
        elif choice == '2':
            game_metrics(record_moves=False)
        elif choice == '3':
            human_vs_ai()
        elif choice == '4':
            print("bye..")
        else:
            print("Invalid choice. Please select either '1' or '2' or '3'.")


start_game()
