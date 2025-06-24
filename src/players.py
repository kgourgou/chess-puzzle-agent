import asyncio
import random
import chess
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import chess.engine

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


def score_move_with_stockfish(move: str, board_fen: str) -> str:
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
        info_best = engine.analyse(board, chess.engine.Limit(time=0.1))
        best_score = info_best["score"].white().score(mate_score=100000)  # centipawns

        # Evaluate given move
        board.push(move)
        info_given = engine.analyse(board, chess.engine.Limit(time=0.1))
        move_score = info_given["score"].white().score(mate_score=100000)
        board.pop()

    score_difference = (move_score if move_score is not None else 0) - (
        best_score if best_score is not None else 0
    )

    thresholds = {
        "great move": 200,
        "good move": 100,
        "equal move": 0,
        "bad move": -100,
        "terrible move": -200,
    }
    for label, threshold in thresholds.items():
        if score_difference >= threshold:
            return f"move {move} is a {label}"


def get_attackers_of_squares(squares: list[str], board_fen: str) -> list[list[str]]:
    """
    Finds all pieces of the opposite color that are attacking specific squares.

    :param squares: The squares to check for attackers (e.g., ['e4', 'd5']).
    :param board_fen: The FEN string representing the current board position.
    :return: A list of lists, each containing squares from which pieces are attacking the corresponding target square.
    """
    board = chess.Board(board_fen)
    attackers_list = []
    for square in squares:
        try:
            target_square = chess.parse_square(square.lower())
            piece_color = board.color_at(target_square)
            attacking_color = (
                not piece_color if piece_color is not None else not board.turn
            )
            attackers = board.attackers(attacking_color, target_square)
            attackers_list.append([chess.square_name(s) for s in attackers])
        except ValueError:
            attackers_list.append(["Error: Invalid square provided."])

    return attackers_list


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


class ChessBM(BaseModel):
    """
    Base model for chess players.
    This model can be extended to include additional fields as needed.
    """

    reasoning: str = Field(
        ..., description="A brief explanation of the reasoning behind the chosen move."
    )
    move: str = Field(..., description="The move in UCI format (e.g., 'e2e4').")


class LLMPlayer(Agent):
    """
    A player that uses a language model to decide on moves in a chess game.
    """

    def __init__(
        self,
        model,
        instructions: str,
        output_type: ChessBM | None = None,
        **kwargs: dict,
    ):
        """
        Initialize the LLMPlayer with a model and instructions.
        :param model: The language model to use for generating moves.
        :param instructions: Instructions for the model to follow.
        :param output_type: The type of output expected from the model, defaults to ChessBM.
        :param kwargs: Additional keyword arguments for the Agent class.
        """
        output_type = output_type or ChessBM

        chess_tools = [
            get_attackers_of_squares,
            # move_is_legal,
            # score_move_with_stockfish,
        ]

        super().__init__(
            model=model,
            instructions=instructions,
            output_type=output_type,
            tools=chess_tools,
            **kwargs,
        )

    def move(
        self,
        board: chess.Board,
        move_limit: int,
        additional_instructions: str | None = None,
        checkmate_retry: bool = False,
        model_settings: dict | None = None,
    ) -> ChessBM:
        """
        Generate a move for the given board using the language model.

        :param board: A chess.Board object representing the current position.
        :param move_limit: The number of moves left in the game.
        :param additional_instructions: Additional instructions for the model, if any.
        :param checkmate_retry: Whether to retry if the move does not result in checkmate.
        :param model_settings: Additional settings for the model, if any.
        :return: A ChessBM object."""
        model_settings = model_settings or {}
        legal_moves = [move.uci() for move in board.legal_moves]
        feedback = ""
        tries = 0
        max_tries = 3  # Limit the number of retries to avoid infinite loops

        while tries < max_tries:
            input_prompt = self.create_instructions_str(
                board,
                legal_moves,
                move_limit,
                ((additional_instructions or "") + " " + (feedback or "")).strip(),
            )

            move = asyncio.run(
                self.run(input_prompt, model_settings=model_settings)
            ).output

            print(f"Model generated move: {move.move} with reasoning: {move.reasoning}")

            if not move:
                print("No move generated by the model, falling back to random player.")
                move = ChessBM(
                    reasoning="Fallback to random move.", move=random_player(board)
                )

            if move.move not in legal_moves:
                print(f"Model generated illegal move: {move.move}")
                feedback += self.retry_if_not_valid(board, move.move)
                tries += 1
                continue

            if checkmate_retry and move_limit == 1:
                feedback += self.retry_if_not_checkmate(board, move.move)
                tries += 1
                continue

            break

        if move.move not in legal_moves:
            return ChessBM(
                reasoning="The model generated an illegal move.",
                move=random_player(board),
            )

        return move

    def retry_if_not_valid(self, board, move) -> str | None:
        if not board.is_legal(chess.Move.from_uci(move)):
            print(f"Move {move} is not valid in the current position ... retrying.")
            return (
                f"Your last move ({move}) is not valid in the current position. "
                "Please try a different legal move."
            )

        return ""

    def retry_if_not_checkmate(self, board, move) -> str | None:
        temp_board = board.copy()
        temp_board.push_uci(move)
        if not temp_board.is_checkmate():
            print(f"Move {move} did not result in checkmate ... retrying.")
            return (
                f"Your last move ({move}) did not result in checkmate. "
                "Please try a different move that delivers checkmate."
            )
        return ""

    def create_instructions_str(
        self,
        board: chess.Board,
        legal_moves: list[chess.Move],
        move_limit: int,
        additional_instructions: str | None = None,
    ) -> str:
        """
        Construct the input prompt for the LLM player.
        :param board: A chess.Board object representing the current position.
        :param legal_moves: A list of legal moves in the current position.
        :param move_limit: The number of moves left in the game.
        """

        prompt = f"""=======CHESS GAME=========
        - Current board position (FEN): {board.fen()}
        - LEGAL MOVES: {", ".join(legal_moves)}
        - MOVES LEFT: {move_limit}
        """

        if additional_instructions:
            prompt += f" - FEEDBACK: {additional_instructions}"

        prompt += "\n=========================\n"

        return prompt


def random_player(board: chess.Board) -> str:
    """
    Given a board, get all of the legal moves in the current position.
    Then pick one of the legal moves at random and return it in UCI format.

    :param board: A chess.Board object representing the current position.
    :return: A string representing the chosen move in UCI format (e.g., 'e2e4').
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise ValueError("No legal moves available in the current position.")
    move = random.choice(legal_moves)
    return move.uci()
