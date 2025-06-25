import asyncio
import typing
import random
import chess
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import chess.engine

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"


# TODO I should move the tools inside the agent class and also have it control a copy of the board.
# THEN I won't have to pass FEN strings around, I can just pass the board object.
def top_5_moves(fen: str) -> list:
    """
    Returns a list of the top 5 legal moves from a given position using Stockfish.

    Args:
        fen (str or chess.Board): The Forsyth-Edwards Notation string of the starting position
                                  or a chess.Board object.

    Returns:
        A list of moves in UCI format. The list is sorted by score in descending order,
        with the top 5 moves returned. If 5 exceeds the number of legal moves, it returns
        all legal moves sorted by score.
    """
    scores = []
    if isinstance(fen, chess.Board):
        board = fen.copy()
    else:
        # If fen is a string, create a board from it
        board = chess.Board(fen)
    try:
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            for move in board.legal_moves:
                board.push(move)
                # Use a very low limit for a quick evaluation
                info = engine.analyse(board, chess.engine.Limit(depth=5))
                score = (
                    info["score"].white().score(mate_score=10000)
                )  # Get a raw number
                scores.append((move.uci(), score))
                board.pop()  # Undo the move
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    k = 5  # Default to top 5 moves
    k = min(k, len(scores))  # Ensure k does not exceed the number of moves
    if k <= 0:
        return []
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    # remove the scores from the output
    return [move for move, _ in sorted_scores]


def is_mate_in_1_moves(move: str, fen: str) -> bool:
    """
    Checks if a move leads to checkmate in exactly 1 from the given position.
    Useful for checking the end of a plan.

    Args:
        move (str): The move in UCI format (e.g., 'e2e4').
        fen (str): The Forsyth-Edwards Notation string of the starting position.

    Returns:
        bool: True if the move leads to checkmate in exactly 1 move, False otherwise.
    """
    board = chess.Board(fen)
    n_moves = 1  # Set the number of moves to check for checkmate
    try:
        uci_move = chess.Move.from_uci(move)
        if uci_move not in board.legal_moves:
            return False
        board.push(uci_move)
        for _ in range(n_moves):
            if board.is_checkmate():
                return True
            # Simulate opponent's best response
            with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
                info = engine.analyse(board, chess.engine.Limit(depth=5))
                if "pv" not in info or not info["pv"]:
                    return False
                best_move = info["pv"][0]
                board.push(best_move)
        return board.is_checkmate()
    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def piece_is_vulnerable(piece_square: str, board_fen: str) -> bool:
    """
    Checks if a piece on a given square is vulnerable to capture by the opponent.
    A vulnerable piece is one that can be attacked by at least one of the opponent's pieces,
    including the king.

    Args:
        piece_square (str): The square of the piece in algebraic notation (e.g., 'e4').
        board_fen (str): The Forsyth-Edwards Notation string of the current board position.

    Returns:
        bool: True if the piece is vulnerable, False otherwise.
    """
    board = chess.Board(board_fen)
    try:
        target_square = chess.parse_square(piece_square.lower())
        piece_color = board.color_at(target_square)
        if piece_color is None:
            return False  # No piece on the square

        attacking_color = not piece_color
        attackers = board.attackers(attacking_color, target_square)

        return len(attackers) > 0
    except ValueError:
        return False  # Invalid square provided


def get_principal_variation(move: str, fen: str) -> dict[str, typing.Any]:
    """
    Analyzes the consequences of a given move from a specific board state.

    This function sets up a board, makes the proposed move, and then asks the
    chess engine to calculate the best continuation (principal variation) for
    the opponent.

    Args:
        move (str): The move in UCI format (e.g., 'e2e4').
        fen (str): The Forsyth-Edwards Notation string of the starting position.

    Returns:
        A dictionary containing the analysis results:
        {
            "best_response": The opponent's best reply to the move.
            "continuation": The full sequence of moves in the principal variation.
            "final_evaluation": The engine's score of the position after the
                                principal variation, from White's perspective.
                                (e.g., +0.5, -3.1, "#2", "-M1").
        }
        Returns an error dictionary if the move is illegal.
    """
    try:
        # Use a `with` statement to ensure the engine process is always closed.
        with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
            # 1. Set up the board from the FEN string.
            board = chess.Board(fen)

            # 2. Check if the move is legal before pushing.
            try:
                uci_move = chess.Move.from_uci(move)
                if uci_move not in board.legal_moves:
                    raise ValueError("Illegal move")
                board.push(uci_move)
            except ValueError:
                return {"error": f"The move '{move}' is illegal in the position."}

            # 3. Analyze the resulting position (it's now the opponent's turn).
            # The `analyse` method returns a dictionary of information.
            info = engine.analyse(board, chess.engine.Limit(depth=5))

            # The principal variation (pv) is a list of moves. The first one
            # is the opponent's best response.
            if "pv" not in info or not info["pv"]:
                # This can happen if the move results in immediate checkmate or stalemate.
                if board.is_checkmate():
                    return {
                        "best_response": None,
                        "continuation": [],
                        "final_evaluation": "Checkmate",
                    }
                else:
                    return {"error": "No principal variation found."}

            principal_variation = info["pv"]

            # 4. Format the output for clarity.
            best_response = principal_variation[0].uci()
            continuation_uci = [m.uci() for m in principal_variation]

            # The score is given from the current player's perspective. We'll
            # convert it to White's perspective for consistency.
            score = info["score"].white()

            if score.is_mate():
                # e.g., Mate(+2) means White can mate in 2. Mate(-1) means Black can mate in 1.
                final_evaluation = f"#{score.mate()}"
            else:
                # Convert centipawn score to standard decimal format.
                final_evaluation = f"{score.cp / 100.0:+.2f}"

            return {
                "best_response": best_response,
                "continuation": continuation_uci,
                "final_evaluation": final_evaluation,
            }

    except FileNotFoundError:
        return {
            "error": f"Stockfish engine not found at '{STOCKFISH_PATH}'. Please check the STOCKFISH_PATH."
        }
    except Exception as e:
        return {"error": f"An unexpected error occurred: {e}"}


def get_best_legal_counter_move_by_opponent(move: str, board_fen: str) -> str:
    """Finds the best counter move to a given move using Stockfish engine.
    The counter move is always a legal move in the current position.

    :param move: Your move in UCI format (e.g., 'e2e4').
    :param board_fen: The FEN string representing the current board position.
    :return: The best counter move in UCI format.
    """
    board = chess.Board(board_fen)
    move = chess.Move.from_uci(move)
    if move not in board.legal_moves:
        return f"Error: Move {move} is not legal in the current position."
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        # Push the move to the board
        board.push(move)
        # Get the best counter move
        info = engine.analyse(board, chess.engine.Limit(time=0.2))
        best_move = info["pv"][0]

    return best_move.uci() if best_move else "No counter move found."


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
        info_best = engine.analyse(board, chess.engine.Limit(time=0.1))
        # best move
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
        "OK move": 0,
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
            # top_3_moves,
            # get_principal_variation
            get_attackers_of_squares,
            # get_best_legal_counter_move_by_opponent,
            # move_is_legal,
            # score_move_against_stockfish,
            # is_mate_in_n_moves,
        ]
        self.cache = {}

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

    def board_description(self, board: str | chess.Board) -> str:
        """
        Generates a human-readable string listing the positions of all pieces on the board,
        along with their legal moves and attacked squares.

        Args:
            board: A python-chess Board object or a FEN string representing the current position.

        Returns:
            A formatted string describing the piece positions, their legal moves, and attacked squares.
        """
        if isinstance(board, str) and board in self.cache:
            return self.cache[board]

        if isinstance(board, str):
            board = chess.Board(board)

        piece_map = {
            chess.PAWN: "Pawn",
            chess.KNIGHT: "Knight",
            chess.BISHOP: "Bishop",
            chess.ROOK: "Rook",
            chess.QUEEN: "Queen",
            chess.KING: "King",
        }

        white_pieces = []
        black_pieces = []

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_name = piece_map[piece.piece_type]
                square_name = chess.square_name(square)
                # Legal moves for this piece
                legal_moves_full = []
                for move in board.legal_moves:
                    if move.from_square == square:
                        if move.promotion:
                            legal_moves_full.append(
                                f"{chess.square_name(move.to_square)}={piece_map[move.promotion][0]}"
                            )
                        else:
                            legal_moves_full.append(chess.square_name(move.to_square))
                if legal_moves_full:
                    moves_str = f"[{', '.join(legal_moves_full)}]"
                else:
                    moves_str = "No legal moves."

                # Attacked squares by this piece
                attacked_squares = [
                    chess.square_name(sq) for sq in board.attacks(square)
                ]
                if attacked_squares:
                    attacks_str = f"Attacks Squares: [{', '.join(attacked_squares)}]"
                else:
                    attacks_str = "Attacks Squares: []"

                desc = (
                    f"- {piece_name} at {square_name}. Legal moves: {moves_str}\n"
                    f"  {attacks_str}"
                )
                if piece.color == chess.WHITE:
                    white_pieces.append(desc)
                else:
                    black_pieces.append(desc)

        white_section = (
            "**White's Pieces:**\n" + "\n".join(white_pieces)
            if white_pieces
            else "**White's Pieces:**\nNone"
        )
        black_section = (
            "**Black's Pieces:**\n" + "\n".join(black_pieces)
            if black_pieces
            else "**Black's Pieces:**\nNone"
        )

        info = f"""{white_section}\n{black_section}"""
        if isinstance(board, str):
            if len(self.cache) < 3:
                self.cache[board] = info
            else:
                self.cache.pop(next(iter(self.cache)))
                self.cache[board] = info

        return info

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
        :param additional_instructions: Additional instructions for the model, if any.
        :return: The constructed input prompt as a string.
        """

        prompt = f"""=======CHESS GAME=========
        - BOARD DESCRIPTION: {self.board_description(board.fen())}
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
