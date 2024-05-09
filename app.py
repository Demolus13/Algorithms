import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu

from TicTacToe import is_full, has_won, print_board, TicTacToe

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
    if 'won' not in st.session_state:
        st.session_state.won = 0
    if 'full' not in st.session_state:
        st.session_state.full = 0

    symbols = {'X': 1, 'O': -1}
    if 'player' not in st.session_state:
        st.session_state.player = symbols[st.selectbox('Choose your symbol', options=['X', 'O'])]
        st.session_state.board = np.zeros((3, 3))
        if st.session_state.player == -1:
            st.session_state.board[2, 2] = 1
        st.session_state.won = 0
        st.session_state.full = 0
        st.session_state.computer = TicTacToe(st.session_state.board, -st.session_state.player)
    else:
        # Get the current symbol
        current_player = symbols[st.selectbox('Choose your symbol', options=['X', 'O'])]

        # If the symbol has changed, reset the board
        if st.session_state.player != current_player:
            st.session_state.player = current_player
            st.session_state.board = np.zeros((3, 3))
            if st.session_state.player == -1:
                st.session_state.board[2, 2] = 1
            st.session_state.won = 0
            st.session_state.full = 0
            st.session_state.computer = TicTacToe(st.session_state.board, -st.session_state.player)

    if st.session_state.player == 1:
        # getting the input data from the user
        col1, col2 = st.columns(2)
        with col1:
            row = st.text_input('Enter Row Number [0-2]')
        with col2:
            col = st.text_input('Enter Column Number [0-2]')

        buttonCol = st.columns(2)
        with buttonCol[0]:
            # player's turn
            if st.button('Play Button') and st.session_state.won == 0:
                row, col = int(row), int(col)
                if 0 <= row and row <=2 and 0 <= col and col <= 2 and st.session_state.board[row, col] == 0:
                    st.session_state.board[row, col] = st.session_state.player
                    st.session_state.won = has_won(st.session_state.board, st.session_state.player)
                    st.session_state.full = is_full(st.session_state.board)

                    # computers turn
                    if st.session_state.won != 1 and st.session_state.full != 1:
                        response = st.session_state.computer.best_move(st.session_state.board, st.session_state.computer.player)
                        st.session_state.board[response[1], response[2]] = st.session_state.computer.player
                        st.session_state.won = -has_won(st.session_state.board, st.session_state.computer.player)
        
        with buttonCol[1]:
            # Reset the board
            if st.button('Reset Button'):
                st.session_state.board = np.zeros((3, 3))
                st.session_state.won = 0
                st.session_state.full = 0

        # display the updated board
        print_board(st.session_state.board)

        # check game states
        if st.session_state.won:
            st.markdown(f"<h3 style='text-align: center;'>{'Player' if st.session_state.won == 1 else 'Computer'} has won!</h3>", unsafe_allow_html=True)
        elif st.session_state.full:
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
            if st.button('Play Button') and st.session_state.won == 0:
                row, col = int(row), int(col)
                if 0 <= row and row <=2 and 0 <= col and col <= 2 and st.session_state.board[row, col] == 0:
                    row, col = int(row), int(col)
                    if st.session_state.board[row, col] == 0:
                        st.session_state.board[row, col] = st.session_state.player
                        st.session_state.won = has_won(st.session_state.board, st.session_state.player)

                        # computers turn
                        if st.session_state.won != 1 and st.session_state.full != 1:
                            response = st.session_state.computer.best_move(st.session_state.board, st.session_state.computer.player)
                            st.session_state.board[response[1], response[2]] = st.session_state.computer.player
                            st.session_state.won = -has_won(st.session_state.board, st.session_state.computer.player)
                            st.session_state.full = is_full(st.session_state.board)
        
        with buttonCol[1]:
            # Reset the board
            if st.button('Reset Button'):
                st.session_state.board = np.zeros((3, 3))
                st.session_state.board[2, 2] = 1
                st.session_state.won = 0
                st.session_state.full = 0

        # display the updated board
        print_board(st.session_state.board)

        # check game states
        if st.session_state.won:
            st.markdown(f"<h3 style='text-align: center;'>{'Player' if st.session_state.won == 1 else 'Computer'} has won!</h3>", unsafe_allow_html=True)
        elif st.session_state.full:
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
    st.write('This is a Linear Regression model for Diabetes Prediction')

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
