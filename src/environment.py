import chess
import gym
import numpy as np
import torch


class ChessEnv(gym.Env):
    """
    Custom Gym environment for chess.

    This environment provides:
    - Standard board initialization and reset functionality.
    - Execution of UCI-formatted moves with legality checks.
    - State representations as tensors suitable for neural networks.
    - Reward calculations based on captures.
    - Game-over detection to signal the end of a training episode.

    """

    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()

    def reset(self):
        """
        Resets the board to the starting position.

        Returns:
            torch.Tensor: The initial state of the board.
        """
        self.board.reset()
        return self.get_state()

    def step(self, action: str):
        """
        Executes a move and updates the board.

        Args:
            action (str): The move in UCI format (e.g., 'e2e4').

        Returns:
            tuple: (next_state, reward, done, info)
        """
        move = chess.Move.from_uci(action)
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            return self.get_state(), -1, True, {"msg": "Illegal move"}

        reward = self.calculate_reward(move)
        done = self.board.is_game_over()
        return self.get_state(), reward, done, {}

    def get_board_state(self, board: chess.Board) -> dict[int, np.ndarray]:
        """
        Generates a representation of the board state.

        Args:
            board (chess.Board): The chess board.

        Returns:
            dict: Maps piece types to 8x8 arrays of their positions.
        """
        state_of_game = {}
        for piece_type in chess.PIECE_TYPES:
            board_state = np.zeros((8, 8), dtype=np.int8)
            for square, piece in board.piece_map().items():
                if piece.piece_type == piece_type:
                    j, i = self.calculate_index_for_square(square)
                    board_state[i][j] = 1 if piece.color else -1
            state_of_game[piece_type] = board_state
        return state_of_game

    def calculate_index_for_square(self, square: int) -> tuple[int, int]:
        """
        Calculates the (row, column) indices for a square.

        Args:
            square (int): The square index (0-63).

        Returns:
            tuple: (column, row).
        """
        return square % 8, square // 8

    def get_state_for_move_action(self, move: chess.Move, board: chess.Board = None) -> torch.Tensor:
        """
        Creates the state representation of a move.

        Args:
            move (chess.Move): The move.
            board (chess.Board, optional): The board for legality check.

        Returns:
            torch.Tensor: A 2x8x8 tensor (from/to squares).
        """
        if board and move not in board.legal_moves:
            raise ValueError("Illegal move")

        from_board = torch.zeros((8, 8), dtype=torch.float32)
        i, j = self.calculate_index_for_square(move.from_square)
        from_board[i][j] = 1

        to_board = torch.zeros((8, 8), dtype=torch.float32)
        i, j = self.calculate_index_for_square(move.to_square)
        to_board[i][j] = 1

        return torch.stack((from_board, to_board), dim=0)

    def get_state(self) -> torch.Tensor:
        """
        Returns the current state of the board.

        Returns:
            torch.Tensor: A tensor of the board state.
        """
        board_state_dict = self.get_board_state(self.board)
        return torch.tensor(list(board_state_dict.values()), dtype=torch.float32)

    def calculate_reward(self, move: chess.Move) -> float:
        """
        Calculates the reward for a move.

        Args:
            move (chess.Move): The move that was executed.

        Returns:
            float: The reward.
        """
        if self.board.is_capture(move):
            captured_piece = self.board.piece_at(move.to_square)
            return self.get_piece_value(captured_piece)
        return 0

    def get_piece_value(self, piece: chess.Piece) -> int:
        """
        Returns the value of a chess piece. This is used to determine how good
        a possible move is.

        Args:
            piece (chess.Piece): The captured piece.

        Returns:
            int: The value of the piece.
        """
        if piece is None:
            return 0
        piece_values = {'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 10}
        return piece_values.get(piece.symbol().upper(), 0)
