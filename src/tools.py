import chess
import chess.engine
import os
from dotenv import load_dotenv
from functools import lru_cache

load_dotenv()

STOCKFISH_PATH = os.getenv("STOCKFISH_PATH")


def is_check(move: str, fen: str) -> bool:
    """
    Checks if a move is a check in the current position.

    :param move: The move in UCI format (e.g., 'e2e4').
    :param fen: The FEN string representing the current board position.
    :return: True if the move is a check, False otherwise.
    """
    try:
        board = chess.Board(fen)
        move_obj = chess.Move.from_uci(move)
        if move_obj not in board.legal_moves:
            return False
        board.push(move_obj)
        is_check = board.is_check()
        board.pop()
        return is_check
    except ValueError:
        return False  # Invalid move format


def shallow_score_of_moves(moves: list[str], board: chess.Board) -> list[float]:
    """
    Given a list of moves, return a list of scores for each move.
    The score is a float representing the quality of the move.
    The higher the score, the better the move.

    :param moves: A list of moves in UCI format (e.g., ['e2e4', 'd2d4']).
    :param board: A chess.Board object representing the current position.
    :return: A list of scores for each move. The score is a float representing the quality of the move.
    The higher the score, the better the move.
    0 means the move is illegal or invalid.
    100000 means the move is a checkmate.
    """
    if not moves:
        return []

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        board = board or chess.Board()
        scores = []
        for move in moves:
            try:
                move_obj = chess.Move.from_uci(move)
                if move_obj in board.legal_moves:
                    board.push(move_obj)
                    info = engine.analyse(board, chess.engine.Limit(time=0.01))
                    score = info["score"].white().score(mate_score=100000)
                    scores.append(score if score is not None else 0)
                    board.pop()
                else:
                    scores.append(0)  # Move is illegal
            except ValueError:
                scores.append(0)  # Invalid move format

    return scores


def get_best_legal_counter_move_by_opponent(board: chess.Board) -> str | None:
    """Finds the best counter move to a given move using Stockfish engine.
    The counter move is always a legal move in the current position.

    :param board: The chess.Board object representing the current board position.
    :return: The best counter move in UCI format.
    """
    if not isinstance(board, chess.Board):
        raise ValueError("The input must be a chess.Board object.")

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        # Get the best counter move
        info = engine.analyse(board, chess.engine.Limit(time=0.001))

        best_move = info.get("pv", [])
        if best_move:
            best_move = best_move[0]
        else:
            best_move = None

    return best_move.uci() if best_move else None


@lru_cache(maxsize=128)
def score_move_against_stockfish(move: str, board_fen: str) -> str:
    """
    Scores a move using Stockfish engine by comparing it to the best move in the position.
    :param move: The move in UCI format (e.g., 'e2e4').
    :param board_fen: The FEN string representing the current board position.
    :return: a float representing the score of the move compared to the best move.

    "great move" > "good move" > "equal move" > "bad move" > "terrible move"

    Even if a move is great, it may not lead to mate soon enough. Always verify plans.
    """
    board = chess.Board(board_fen)
    move = chess.Move.from_uci(move)
    if move not in board.legal_moves:
        return f"Error: Move {move} is not legal in the current position."

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        # Evaluate best move
        info_best = engine.analyse(board, chess.engine.Limit(time=0.5))
        # best move
        best_score = info_best["score"].white().score(mate_score=100000)  # centipawns

        # Evaluate given move
        board.push(move)
        info_given = engine.analyse(board, chess.engine.Limit(time=0.5))
        move_score = info_given["score"].white().score(mate_score=100000)
        board.pop()

    score_difference = (move_score if move_score is not None else 0) - (
        best_score if best_score is not None else 0
    )

    return score_difference


@lru_cache(maxsize=128)
def move_is_legal(move: str, board_fen: str) -> bool:
    """
    Checks if a move is legal in the given board position.

    :param move: The move in UCI format (e.g., 'e2e4').
    :param board_fen: The FEN string representing the current board position.
    :return: True if the move is legal, False otherwise.
    """
    board = chess.Board(board_fen)
    try:
        move_obj = chess.Move.from_uci(move)
        return move_obj in board.legal_moves
    except ValueError:
        return False


def are_sequences_checkmate(moves: list[list[str]], board_fen: str) -> list[bool]:
    """
    Checks if a sequence of moves from a given board position results in checkmate.
    The sequence should include moves for both White and Black.

    :param moves: A list of lists, where each inner list contains moves in UCI format.
                  Each inner list represents a sequence of moves (e.g., [['e2e4', 'e7e5'], ['d2d4', 'e5d4']]).
    :param board_fen: The FEN string for the starting board position.
    :return: A list of booleans indicating whether each sequence results in checkmate.
    """
    results = []
    for move_sequence in moves:
        temp_board = chess.Board(board_fen)
        try:
            for move_uci in move_sequence:
                move = chess.Move.from_uci(move_uci)
                if move in temp_board.legal_moves:
                    temp_board.push(move)
                else:
                    results.append(False)
                    break
            else:
                results.append(temp_board.is_checkmate())
        except ValueError:
            results.append(False)
    return results
