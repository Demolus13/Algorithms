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

n_board = 3

def board_encoding(board: np.ndarray) -> int:
    """
    board: np.ndarray: Playing board as an array

    Return: int: board encoding
    """

    encode = 0
    power = 1
    for row in range(n_board):
        for col in range(n_board):
            if board[row, col] == 1:
                encode += power
            elif board[row, col] == -1:
                encode += 2*power
            
            power *= n_board

    return encode

def print_board(board: np.ndarray):
    """
    board: np.ndarray: Playing board as an array
    """

    symbols = {0: '--', 1: 'X', -1: 'O'}
    for i in range(3):
        cols = st.columns(3)
        for j in range(3):
            cols[j].markdown(f"<h3 style='text-align: center;'>{symbols[board[i][j]]}</h3>", unsafe_allow_html=True)

def is_full(board: np.ndarray) -> bool:
    """
    board: np.ndarray: Playing board as an array

    Return: bool: if the board is full
    """

    for row in range(n_board):
        for col in range(n_board):
            if board[row, col] == 0:
                return False
    return True

def has_won(board: np.ndarray, player: int) -> bool:
    """
    board: np.ndarray: Playing board as an array
    player: int: Numeric encoding

    Return: bool: if player has won
    """

    # check for vertical
    for row in range(n_board):
        for col in range(n_board):
            if board[row, col] != player:
                break
        else:
            return True
        
    # check for horizontal
    for col in range(n_board):
        for row in range(n_board):
            if board[row, col] != player:
                break
        else:
            return True
        
    # check for diagonal 1
    for row in range(n_board):
        if board[row, row] != player:
            break
    else:
        return True
    
    # check for diagonal 2
    for row in range(n_board):
        if board[row, n_board - row - 1] != player:
            return False
    else:
        return True
    
class TicTacToe():
    """
    An Optimal TicTacToe player
    """

    def __init__(self, board: np.ndarray, player: int):
        """
        board: np.ndarray: Playing board as an array
        player: int: Numeric encoding
        """

        self.board = board
        self.player = player
        self.computed_moves = np.zeros((19681, 4), dtype=np.int32)

    def best_move(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        board: np.ndarray: Playing board as an array

        Return: np.ndarray: seen, row, column and score
        """

        assert ~is_full(board)
        assert ~has_won(board, player)
        assert ~has_won(board, -player)

        no_candidate = 1
        candidate = np.zeros((1, 4), dtype=np.int32)

        encode = board_encoding(board)
        if self.computed_moves[encode][0]:
            return self.computed_moves[encode]
        
        for row in range(n_board):
            for col in range(n_board):
                if board[row, col] == 0:
                    board[row, col] = player
                    if has_won(board, player):
                        board[row, col] = 0
                        candidate = np.array([1, row, col, 1])
                        self.computed_moves[encode] = candidate
                        return candidate
                    board[row, col] = 0

        for row in range(n_board):
            for col in range(n_board):
                if board[row, col] == 0:
                    board[row, col] = player
                    if is_full(board):
                        board[row, col] = 0
                        candidate = np.array([1, row, col, 0])
                        self.computed_moves[encode] = candidate
                        return candidate
                    
                    response = self.best_move(board, -player)
                    board[row, col] = 0
                    if response[3] == -1:
                        candidate = np.array([1, row, col, 1])
                        self.computed_moves[encode] = candidate
                        return candidate
                    elif response[3] == 0:
                        candidate = np.array([1, row, col, 0])
                        self.computed_moves[encode] = candidate
                        no_candidate = 0
                    elif no_candidate:
                        candidate = np.array([1, row, col, -1])
                        self.computed_moves[encode] = candidate
                        no_candidate = 0
        
        self.computed_moves[encode] = candidate
        return candidate