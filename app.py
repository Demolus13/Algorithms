import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

import TicTacToe
import Connect4

# Set page configuration
st.set_page_config(page_title="Algorithmic Games",
                   layout="centered")

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Games',
                           [
                               'TicTacToe',
                               'Connect4',
                               'Sudoku',
                               'Game of Sim',
                            ],
                           default_index=0)


# TicTacToe Page
if selected == 'TicTacToe':

    # page title
    st.title('TicTacToe')
    st.write('Algorithm to play the optimal move in a TicTacToe game')

    # Initialize session_state
    if 'won1' not in st.session_state:
        st.session_state.won1 = 0
    if 'full1' not in st.session_state:
        st.session_state.full1 = 0

    symbols = {'X': 1, 'O': -1}
    if 'player1' not in st.session_state:
        st.session_state.player1 = symbols[st.selectbox('Choose your symbol', options=['X', 'O'])]
        st.session_state.board1 = np.zeros((3, 3))
        if st.session_state.player1 == -1:
            st.session_state.board1[2, 2] = 1
        st.session_state.won1 = 0
        st.session_state.full1 = 0
        st.session_state.computer1 = TicTacToe.TicTacToe(st.session_state.board1, -st.session_state.player1)
    else:
        # Get the current symbol
        current_player = symbols[st.selectbox('Choose your symbol', options=['X', 'O'])]

        # If the symbol has changed, reset the board
        if st.session_state.player1 != current_player:
            st.session_state.player1 = current_player
            st.session_state.board1 = np.zeros((3, 3))
            if st.session_state.player1 == -1:
                st.session_state.board1[2, 2] = 1
            st.session_state.won1 = 0
            st.session_state.full1 = 0
            st.session_state.computer1 = TicTacToe.TicTacToe(st.session_state.board1, -st.session_state.player1)

    if st.session_state.player1 == 1:
        # getting the input data from the user
        col1, col2 = st.columns(2)
        with col1:
            row = st.text_input('Enter Row Number [0-2]')
        with col2:
            col = st.text_input('Enter Column Number [0-2]')

        buttonCol = st.columns(2)
        with buttonCol[0]:
            # player's turn
            if st.button('Play Button') and st.session_state.won1 == 0:
                row, col = int(row), int(col)
                if 0 <= row and row <=2 and 0 <= col and col <= 2 and st.session_state.board1[row, col] == 0:
                    st.session_state.board1[row, col] = st.session_state.player1
                    st.session_state.won1 = TicTacToe.has_won(st.session_state.board1, st.session_state.player1)
                    st.session_state.full1 = TicTacToe.is_full(st.session_state.board1)

                    # computers turn
                    if st.session_state.won1 != 1 and st.session_state.full1 != 1:
                        response = st.session_state.computer1.best_move(st.session_state.board1, st.session_state.computer1.player)
                        st.session_state.board1[response[1], response[2]] = st.session_state.computer1.player
                        st.session_state.won1 = -TicTacToe.has_won(st.session_state.board1, st.session_state.computer1.player)
        
        with buttonCol[1]:
            # Reset the board
            if st.button('Reset Button'):
                st.session_state.board1 = np.zeros((3, 3))
                st.session_state.won1 = 0
                st.session_state.full1 = 0

        # display the updated board
        TicTacToe.print_board(st.session_state.board1)

        # check game states
        if st.session_state.won1:
            st.markdown(f"<h3 style='text-align: center;'>{'Player' if st.session_state.won1 == 1 else 'Computer'} has won!</h3>", unsafe_allow_html=True)
        elif st.session_state.full1:
            st.markdown(f"<h3 style='text-align: center;'>It's a Draw!</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center;'>Ongoing game ...</h3>", unsafe_allow_html=True)
    else:
        # getting the input data from the user
        col1, col2 = st.columns(2)
        with col1:
            row = st.text_input('Enter Row Number [0-2]')
        with col2:
            col = st.text_input('Enter Column Number [0-2]')

        buttonCol = st.columns(2)
        with buttonCol[0]:
            # player's turn
            if st.button('Play Button') and st.session_state.won1 == 0:
                row, col = int(row), int(col)
                if 0 <= row and row <=2 and 0 <= col and col <= 2 and st.session_state.board1[row, col] == 0:
                    st.session_state.board1[row, col] = st.session_state.player1
                    st.session_state.won1 = TicTacToe.has_won(st.session_state.board1, st.session_state.player1)

                    # computers turn
                    if st.session_state.won1 != 1:
                        response = st.session_state.computer1.best_move(st.session_state.board1, st.session_state.computer1.player)
                        st.session_state.board1[response[1], response[2]] = st.session_state.computer1.player
                        st.session_state.won1 = -TicTacToe.has_won(st.session_state.board1, st.session_state.computer1.player)
                        st.session_state.full1 = TicTacToe.is_full(st.session_state.board1)
        
        with buttonCol[1]:
            # Reset the board
            if st.button('Reset Button'):
                st.session_state.board1 = np.zeros((3, 3))
                st.session_state.board1[2, 2] = 1
                st.session_state.won1 = 0
                st.session_state.full1 = 0

        # display the updated board
        TicTacToe.print_board(st.session_state.board1)

        # check game states
        if st.session_state.won1:
            st.markdown(f"<h3 style='text-align: center;'>{'Player' if st.session_state.won1 == 1 else 'Computer'} has won!</h3>", unsafe_allow_html=True)
        elif st.session_state.full1:
            st.markdown(f"<h3 style='text-align: center;'>It's a Draw!</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center;'>Ongoing game ...</h3>", unsafe_allow_html=True)

    # code block
    st.title('Notebook')
    code = '''
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

        # check for horizontal
        for row in range(n_board):
            for col in range(n_board):
                if board[row, col] != player:
                    break
            else:
                return True
            
        # check for vertical
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
    '''
    st.code(code, language='python')
    code = '''
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
            player: int: Numeric encoding
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
    '''
    st.code(code, language='python')


# Connect4' Page
if selected == 'Connect4':

    # page title
    st.title('Connect4')
    st.write('Algorithm to play the optimal move in a Connect4 game')

    # Initialize session_state
    if 'won2' not in st.session_state:
        st.session_state.won2 = 0
    if 'full2' not in st.session_state:
        st.session_state.full2 = 0

    # Computer player hyperparameters
    depth = st.slider('Thinking Depth', min_value=1, max_value=15, value=6)

    symbols = {'R': 1, 'B': -1}
    if 'player2' not in st.session_state:
        st.session_state.player2 = symbols[st.selectbox('Choose your symbol', options=['R', 'B'])]
        st.session_state.board2 = np.zeros((4, 5))
        if st.session_state.player2 == -1:
            st.session_state.board2[3, 0] = 1
        st.session_state.won2 = 0
        st.session_state.full2 = 0
        st.session_state.computer2 = Connect4.Connect4(st.session_state.board2, -st.session_state.player2)
    else:
        # Get the current symbol
        current_player = symbols[st.selectbox('Choose your symbol', options=['R', 'B'])]

        # If the symbol has changed, reset the board
        if st.session_state.player2 != current_player:
            st.session_state.player2 = current_player
            st.session_state.board2 = np.zeros((4, 5))
            if st.session_state.player2 == -1:
                st.session_state.board2[3, 0] = 1
            st.session_state.won2 = 0
            st.session_state.full2 = 0
            st.session_state.computer2 = Connect4.Connect4(st.session_state.board2, -st.session_state.player2)

    if st.session_state.player2 == 1:
        # getting the input data from the user
        col1 = st.columns(1)
        with col1[0]:
            col = st.text_input('Enter Column Number [0-4]')

        buttonCol = st.columns(2)
        with buttonCol[0]:
            # player's turn
            if st.button('Play Button') and st.session_state.won2 == 0:
                col = int(col)
                if 0 <= col and col <= 4 and st.session_state.board2[0, col] == 0:
                    Connect4.drop_piece(st.session_state.board2, col, st.session_state.player2)
                    st.session_state.won2 = Connect4.has_won(st.session_state.board2, st.session_state.player2)

                    # computers turn
                    if st.session_state.won2 != 1:
                        response = st.session_state.computer2.best_move(st.session_state.board2, st.session_state.computer2.player, depth=depth)
                        Connect4.drop_piece(st.session_state.board2, response[0], st.session_state.computer2.player)
                        st.session_state.won2 = -Connect4.has_won(st.session_state.board2, st.session_state.computer2.player)
                        st.session_state.full2 = Connect4.is_full(st.session_state.board2)
        
        with buttonCol[1]:
            # Reset the board
            if st.button('Reset Button'):
                st.session_state.board2 = np.zeros((4, 5))
                st.session_state.won2 = 0
                st.session_state.full2 = 0

        # display the updated board
        Connect4.print_board(st.session_state.board2)

        # check game states
        if st.session_state.won2:
            st.markdown(f"<h3 style='text-align: center;'>{'Player' if st.session_state.won2 == 1 else 'Computer'} has won!</h3>", unsafe_allow_html=True)
        elif st.session_state.full2:
            st.markdown(f"<h3 style='text-align: center;'>It's a Draw!</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center;'>Ongoing game ...</h3>", unsafe_allow_html=True)
    else:
        # getting the input data from the user
        col1 = st.columns(1)
        with col1[0]:
            col = st.text_input('Enter Column Number [0-4]')

        buttonCol = st.columns(2)
        with buttonCol[0]:
            # player's turn
            if st.button('Play Button') and st.session_state.won2 == 0:
                col = int(col)
                if 0 <= col and col <= 4 and st.session_state.board2[0, col] == 0:
                    Connect4.drop_piece(st.session_state.board2, col, st.session_state.player2)
                    st.session_state.won2 = Connect4.has_won(st.session_state.board2, st.session_state.player2)
                    st.session_state.full2 = Connect4.is_full(st.session_state.board2)

                    # computers turn
                    if st.session_state.won2 != 1 and st.session_state.full2 != 1:
                        response = st.session_state.computer2.best_move(st.session_state.board2, st.session_state.computer2.player)
                        Connect4.drop_piece(st.session_state.board2, response[0], st.session_state.computer2.player)
                        st.session_state.won2 = -Connect4.has_won(st.session_state.board2, st.session_state.computer2.player)
        
        with buttonCol[1]:
            # Reset the board
            if st.button('Reset Button'):
                st.session_state.board2 = np.zeros((4, 5))
                st.session_state.board2[3, 0] = 1
                st.session_state.won2 = 0
                st.session_state.full2 = 0

        # display the updated board
        Connect4.print_board(st.session_state.board2)

        # check game states
        if st.session_state.won2:
            st.markdown(f"<h3 style='text-align: center;'>{'Player' if st.session_state.won2 == 1 else 'Computer'} has won!</h3>", unsafe_allow_html=True)
        elif st.session_state.full2:
            st.markdown(f"<h3 style='text-align: center;'>It's a Draw!</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='text-align: center;'>Ongoing game ...</h3>", unsafe_allow_html=True)

    # code block
    st.title('Notebook')
    code = '''
    """
    Utilities for the game
    """

    h_board = 4
    w_board = 5

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
    '''
    st.code(code, language='python')
    code = '''
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
    '''
    st.code(code, language='python')


# Sudoku Page
if selected == 'Sudoku':

    # page title
    st.title('Sudoku')
    st.write('This is a Decison Tree model for Diabetes Prediction')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DTreePedigreeFunction = st.text_input('DTree Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DTreePedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = 1
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(f"Prediction: {diab_diagnosis}")

    # code block
    st.title('Notebook')
    code = '''
    def hello():
        print("Hello, Streamlit!")
    '''
    st.code(code, language='python')


# Game of Sim Page
if selected == 'Game of Sim':

    # page title
    st.title('Game of Sim')
    st.write('This is a Decison Tree model for Diabetes Prediction')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DTreePedigreeFunction = st.text_input('DTree Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction
    if st.button('Predict'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DTreePedigreeFunction, Age]
        user_input = [float(x) for x in user_input]
        diab_prediction = 1
        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(f"Prediction: {diab_diagnosis}")

    # code block
    st.title('Notebook')
    code = '''
    def hello():
        print("Hello, Streamlit!")
    '''
    st.code(code, language='python')
