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

n_board = 9

def print_board(board: np.ndarray):
    """
    board: np.ndarray: Sudoku board as an array
    """

    for i in range(n_board):
        cols = st.columns(n_board)
        for j in range(n_board):
            cols[j].markdown(f"<h5 style='text-align: center;'>{str(board[i, j]) if board[i, j] else '.'}</h5>", unsafe_allow_html=True)

def is_full(board: np.ndarray) -> np.ndarray:
    """
    board: np.ndarray: Sudoku board as an array

    Return: np.ndarray: is_full, row, and column
    """

    for row in range(n_board):
        for col in range(n_board):
            if board[row, col] == 0:
                return [0, row, col]
    return [1, -1, -1]

def is_valid(board: np.ndarray) -> bool:
    """
    board: np.ndarray: Sudoku board as an array

    Return: bool: if the board is a valid sudoku board
    """

    # check for horizontal
    for row in range(n_board):
        counts = np.bincount(board[row, :])
        if any(count > 1 for count in counts[1:]):
            return False
        
    # check for vertical
    for col in range(n_board):
        counts = np.bincount(board[:, col])
        if any(count > 1 for count in counts[1:]):
            return False
        
    # check for boxes
    for row in range(0, n_board, 3):
        for col in range(0, n_board, 3):
            box = board[row:row+3, col:col+3].flatten()
            counts = np.bincount(box)
            if any(count > 1 for count in counts[1:]):
                return False
    
    return True

class Sudoku():
    """
    An Sudoku Solver
    """

    def __init__(self, board: np.ndarray):
        """
        board: np.ndarray: Sudoku board as an array
        """

        self.board = board

    def solve(self) -> np.ndarray:
        """
        Return: np.ndarray: Solved Sudoku board
        """

        # solved sudoku board
        find = is_full(self.board)
        if find[0]:
            return True
        else:
            row, col = find[1:]

        for i in range(1, 10):
            self.board[row, col] = i
            if is_valid(self.board):
                
                if self.solve():
                    return True
                
            self.board[row, col] = 0

        return False
