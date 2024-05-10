"""
Importing the necessary libraries
"""
import numpy as np
import streamlit as st

# Remove all the warnings
import warnings
warnings.filterwarnings('ignore')

"""
Utilities for the game
"""

h_board = 4
w_board = 5

def print_board(board: np.ndarray):
    """
    board: np.ndarray: Playing board as an array
    """

    symbols = {0: '--', 1: 'R', -1: 'B'}
    for i in range(h_board):
        cols = st.columns(w_board)
        for j in range(w_board):
            cols[j].markdown(f"<h3 style='text-align: center;'>{symbols[board[i][j]]}</h3>", unsafe_allow_html=True)

def is_full(board: np.ndarray) -> bool:
    """
    board: np.ndarray: Playing board as an array

    Return: bool: if the board is full
    """

    for col in range(w_board):
        if board[0, col] == 0:
            return False
    return True

def score(board: np.ndarray, player: int) -> int:
    """
    board: np.ndarray: Playing board as an array
    player: int: Player score

    Return: int: Score of the player
    """

    score = 15
    for row in range(h_board):
        for col in range(w_board):
            if board[row, col] == player:
                score -= 1
    return score

def drop_piece(board: np.ndarray, col: int, player: int) -> int:
    """
    board: np.ndarray: Playing board as an array
    col: int: Column to drop the piece
    player: int: Whose piece to drop

    Return: int: row index of dropped piece
    """

    for row in range(h_board-1, -1, -1):
        if board[row, col] == 0:
            board[row, col] = player
            return row

def has_won(board: np.ndarray, player: int) -> bool:
    """
    board: np.ndarray: Playing board as an array
    player: int: Numeric encoding

    Return: bool: if player has won
    """

    # check for horizontal
    for row in range(h_board):
        for col in range(2):
            if board[row, col] == player and board[row, col+1] == player and board[row, col+2] == player and board[row, col+3] == player:
                return True
        
    # check for vertical
    for col in range(w_board):
        for row in range(h_board):
            if board[row, col] != player:
                break
        else:
            return True
        
    # check for diagonal 1
    for col in range(2):
        if board[0, col] == player and board[1, col+1] == player and board[2, col+2] == player and board[3, col+3] == player:
            return True
    
    # check for diagonal 2
    for col in range(3, w_board):
        if board[0, col] == player and board[1, col-1] == player and board[2, col-2] == player and board[3, col-3] == player:
            return True
        
    return False

class Connect4():
    """
    An Optimal Connect4 player
    """

    def __init__(self, board: np.ndarray, player: int):
        """
        board: np.ndarray: Playing board as an array
        player: int: Numeric encoding
        """

        self.board = board
        self.player = player

    def best_move(self, board: np.ndarray, player: int, depth: int = 6, alpha: int = -np.inf, beta: int = np.inf) -> np.ndarray:
        """
        board: np.ndarray: Playing board as an array
        player: int: Numeric encoding
        depth: int: Search depth of the computer
        alpha: int: maximum possible score
        beta: int: minimum possible score
        """

        assert ~is_full(board)
        assert ~has_won(board, player)
        assert ~has_won(board, -1*player)

        candidate = np.array([-1, -15 if player == self.player else 15])

        if depth == 0:
            candidate = np.array([-1, 0])
            return candidate

        for col in range(w_board):
            if board[0, col] == 0:
                row = drop_piece(board, col, player)
                if has_won(board, player):
                    board[row, col] = 0
                    points = score(board, player) if player == self.player else -score(board, player)
                    candidate = np.array([col, points])
                    return candidate
                board[row, col] = 0

        for col in range(w_board):
            if board[0, col] == 0:
                row = drop_piece(board, col, player)
                if is_full(board):
                    board[row, col] = 0
                    candidate = np.array([col, 0])
                    return candidate
                
                response = self.best_move(board, -1*player, depth-1, alpha, beta)
                board[row, col] = 0
                if player == self.player:
                    if response[1] > candidate[1]:
                        candidate[0] = col
                        candidate[1] = response[1]
                    alpha = alpha if alpha > response[1] else response[1]
                else:
                    if response[1] < candidate[1]:
                        candidate[0] = col
                        candidate[1] = response[1]
                    beta = beta if beta < response[1] else response[1]

                if beta <= alpha:
                    break
        
        return candidate