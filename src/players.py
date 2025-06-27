import asyncio
import typing
import random
import chess
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import chess.engine
from pydantic_ai.models.openai import OpenAIModel

from dotenv import load_dotenv

load_dotenv()

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

model_name = "anthropic/claude-3-7-sonnet-20250219"
lm = OpenAIModel(
    model_name=model_name,
    provider="openrouter",
)


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

    legal_moves = [m.uci() for m in board.legal_moves]
    if move not in legal_moves:
        return f"Error: Move {move} is not legal in the current position."

    move = chess.Move.from_uci(move)

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        # Push the move to the board
        board.push(move)
        # Get the best counter move
        info = engine.analyse(board, chess.engine.Limit(time=0.2))

        best_move = info.get("pv", [])
        if best_move:
            best_move = best_move[0]
        else:
            best_move = None

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
            # get_attackers_of_squares,
            # are_sequences_checkmate,
            get_best_legal_counter_move_by_opponent,
        ]

        # simulate_and_evaluate]
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

            if checkmate_retry:
                feedback += self.retry_if_not_checkmate(board, move.move, move_limit)
                if feedback:  # Only retry if there was feedback (i.e., move didn't achieve puzzle goal)
                    tries += 1
                    continue

            break

        if move.move not in legal_moves:
            return ChessBM(
                reasoning="The model generated an illegal move.",
                move=random_player(board),
            )

        return move

    def retry_if_not_valid(self, board: chess.Board, move: str) -> str | None:
        try:
            move_obj = chess.Move.from_uci(move)
            if not board.is_legal(move_obj):
                print(f"Move {move} is not valid in the current position ... retrying.")

                # Provide specific feedback about why the move is invalid
                from_square = move_obj.from_square
                to_square = move_obj.to_square
                piece = board.piece_at(from_square)
                target_piece = board.piece_at(to_square)

                feedback = f"Your move {move} is illegal. "

                if piece is None:
                    feedback += (
                        f"There is no piece on {chess.square_name(from_square)}. "
                    )
                elif piece.color != board.turn:
                    feedback += f"The piece on {chess.square_name(from_square)} belongs to the opponent. "
                elif target_piece and target_piece.color == piece.color:
                    feedback += f"You cannot capture your own piece on {chess.square_name(to_square)}. "
                else:
                    # Use piece_map to get the piece name
                    piece_map = {
                        chess.PAWN: "Pawn",
                        chess.KNIGHT: "Knight",
                        chess.BISHOP: "Bishop",
                        chess.ROOK: "Rook",
                        chess.QUEEN: "Queen",
                        chess.KING: "King",
                    }
                    piece_name = piece_map.get(piece.piece_type, "Unknown piece")
                    feedback += f"The {piece_name} on {chess.square_name(from_square)} cannot move to {chess.square_name(to_square)}. "

                # List some legal moves for that piece if it exists
                if piece and piece.color == board.turn:
                    legal_moves_for_piece = [
                        m.uci()
                        for m in board.legal_moves
                        if m.from_square == from_square
                    ]
                    if legal_moves_for_piece:
                        feedback += f"Legal moves for this piece: {', '.join(legal_moves_for_piece[:5])}{'...' if len(legal_moves_for_piece) > 5 else ''}. "

                return feedback
        except ValueError:
            print(f"Move {move} has invalid UCI format ... retrying.")
            return (
                f"Your move '{move}' is not in valid UCI format (e.g., 'e2e4'). "
                "Please use the format 'from_square to_square' like 'e2e4' or 'g1f3'."
            )

        return ""

    def retry_if_not_checkmate(self, board, move, move_limit) -> str | None:
        temp_board = board.copy()
        temp_board.push_uci(move)

        # For 1-move puzzles, we need immediate checkmate
        if move_limit == 1:
            if not temp_board.is_checkmate():
                print(f"Move {move} did not result in checkmate ... retrying.")
                return (
                    f"Your last move ({move}) did not result in checkmate. "
                    "This is a mate-in-1 puzzle - you need to deliver checkmate immediately."
                )

        # For multi-move puzzles, we need to check if the move is forcing and leads toward mate
        else:
            # Check if the move gives check (usually required in multi-move puzzles)
            if not temp_board.is_check():
                print(
                    f"Move {move} doesn't give check in a multi-move puzzle ... retrying."
                )
                return (
                    f"Your last move ({move}) doesn't give check. "
                    f"In mate-in-{move_limit} puzzles, moves should typically be forcing (give check)."
                )

            # For mate-in-2+ puzzles, check if we're on the right track
            # by seeing if opponent's best response still allows mate
            if move_limit == 2:
                # After our move and opponent's best response, can we still mate in 1?
                try:
                    counter_move = get_best_legal_counter_move_by_opponent(
                        move, board.fen()
                    )
                    if counter_move and counter_move != "No counter move found.":
                        # Simulate opponent's response
                        temp_board2 = temp_board.copy()
                        temp_board2.push_uci(counter_move)

                        # Check if we have mate in 1 from resulting position
                        mate_in_1_found = False
                        for our_next_move in temp_board2.legal_moves:
                            test_board = temp_board2.copy()
                            test_board.push(our_next_move)
                            if test_board.is_checkmate():
                                mate_in_1_found = True
                                break

                        if not mate_in_1_found:
                            print(
                                f"Move {move} doesn't lead to mate in 2 after opponent's best response ... retrying."
                            )
                            return (
                                f"Your last move ({move}) gives check but after opponent's best response ({counter_move}), "
                                f"you don't have mate in 1. Look for more forcing moves."
                            )
                except:
                    pass  # If tool fails, don't provide this feedback

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

        # --- Enhanced tactical information ---
        tactical_info = []

        # King safety analysis
        for color, name in [(chess.WHITE, "White"), (chess.BLACK, "Black")]:
            king_square = board.king(color)
            if king_square is not None:
                king_status = f"{name} king is at {chess.square_name(king_square)}."
                if board.is_checkmate() and board.turn == color:
                    king_status += " The king is in checkmate."
                elif board.is_stalemate() and board.turn == color:
                    king_status += " The king is stalemated."
                elif board.is_check() and board.turn == color:
                    king_status += " The king is currently in check."
                    # Count escape squares
                    king_attacks = board.attacks(king_square)
                    escape_squares = 0
                    for escape_sq in king_attacks:
                        temp_board = board.copy()
                        try:
                            temp_board.push(chess.Move(king_square, escape_sq))
                            if not temp_board.is_check():
                                escape_squares += 1
                        except:
                            pass
                    king_status += (
                        f" King has {escape_squares} possible escape squares."
                    )
                else:
                    king_status += " The king is not currently in check."

                # Check if king is trapped by own pieces
                if (
                    color == chess.BLACK
                ):  # Focus on black king since we're solving as white
                    adjacent_squares = [
                        king_square + delta
                        for delta in [-9, -8, -7, -1, 1, 7, 8, 9]
                        if 0 <= king_square + delta < 64
                        and abs((king_square % 8) - ((king_square + delta) % 8)) <= 1
                    ]
                    blocked_by_own = sum(
                        1
                        for sq in adjacent_squares
                        if board.piece_at(sq) and board.piece_at(sq).color == color
                    )
                    if blocked_by_own >= 5:
                        king_status += (
                            " TACTICAL NOTE: King is heavily restricted by own pieces!"
                        )

                # Castling rights
                if color == chess.WHITE:
                    if board.has_kingside_castling_rights(
                        chess.WHITE
                    ) or board.has_queenside_castling_rights(chess.WHITE):
                        king_status += " Castling is still possible."
                else:
                    if board.has_kingside_castling_rights(
                        chess.BLACK
                    ) or board.has_queenside_castling_rights(chess.BLACK):
                        king_status += " Castling is still possible."

                tactical_info.append(king_status)
            else:
                tactical_info.append(f"{name} king is not on the board.")

        # Look for common tactical patterns
        pattern_info = []

        # Check for back rank weakness
        black_king_square = board.king(chess.BLACK)
        if black_king_square is not None:
            king_rank = chess.square_rank(black_king_square)
            if king_rank == 7:  # Black king on back rank
                # Check if blocked by own pawns
                files_blocked = 0
                for file_idx in range(
                    max(0, chess.square_file(black_king_square) - 1),
                    min(8, chess.square_file(black_king_square) + 2),
                ):
                    pawn_square = chess.square(file_idx, 6)  # 7th rank for black pawns
                    if board.piece_at(pawn_square) and board.piece_at(
                        pawn_square
                    ) == chess.Piece(chess.PAWN, chess.BLACK):
                        files_blocked += 1
                if files_blocked >= 2:
                    pattern_info.append(
                        "TACTICAL PATTERN: Black king trapped on back rank by own pawns - look for back rank mate!"
                    )

        # Check for pieces that can give discovered check
        white_pieces_squares = [
            sq
            for sq in chess.SQUARES
            if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE
        ]
        for piece_sq in white_pieces_squares:
            piece = board.piece_at(piece_sq)
            if piece and piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
                # Check if moving another white piece could create discovered check
                for other_sq in white_pieces_squares:
                    if other_sq != piece_sq:
                        temp_board = board.copy()
                        temp_board.remove_piece_at(other_sq)
                        if temp_board.is_check():
                            pattern_info.append(
                                f"TACTICAL OPPORTUNITY: Moving piece from {chess.square_name(other_sq)} creates discovered check!"
                            )
                            break

        if pattern_info:
            tactical_info.extend(pattern_info)

        info = f"""{white_section}\n{black_section}\nTactical information:\n- {"-".join(tactical_info)}"""
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


def simulate_and_evaluate(move: str, fen: str) -> str:
    """
    Simulates a move on the given FEN, analyzes the resulting position with Stockfish,
    and returns commentary on the move.

    This simulation may not 100% match the opponent's responses, but it is against the
    best continuation according to Stockfish. It provides an evaluation of the move,
    whether it leads to check, and the best continuation from that position.

    Args:
        move (str): The move in UCI format (e.g., 'e2e4').
        fen (str): The Forsyth-Edwards Notation string of the starting position
    """

    pv_depth = 3
    board = chess.Board(fen)
    legal_moves = [m.uci() for m in board.legal_moves]
    if move not in legal_moves:
        return f"Error: Move {move} is not legal in the current position."

    move_obj = chess.Move.from_uci(move)
    board.push(move_obj)
    is_check = board.is_check()

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=pv_depth), multipv=1)[0]
        score = info["score"].white()
        # Handle mate scores
        if score.is_mate():
            evaluation = f"#{score.mate()}"
        else:
            evaluation = f"{score.score(mate_score=100000) / 100:.1f}"

        # Get the principal variation (PV)
        pv_moves = info.get("pv", [])
        continuation = [move.uci() for move in pv_moves]

    # --- LLM comment generation (placeholder) ---
    # You can replace this with a call to your LLM for richer commentary.
    if score.is_mate() and score.mate() > 0:
        comment = f"This move leads to a forced mate in {score.mate()}."
    elif score.is_mate() and score.mate() < 0:
        comment = (
            f"This move allows a forced mate for the opponent in {abs(score.mate())}."
        )
    elif float(evaluation) > 2:
        comment = "This move gives White a decisive advantage."
    elif float(evaluation) < -2:
        comment = "This move gives Black a decisive advantage."
    else:
        comment = "This is a quiet move. The position remains balanced."

    # commentary = Agent(
    #     model=lm,
    #     instructions="Generate a tactical and concise commentary on the move proposed based on the evaluation and continuation. Your objective is to spell out what you get as input from stockfish and provide a clear, tactical analysis of the move. Do not provide any additional information or context. Mention also the continuation and whether it leads to checkmate.",
    #     output_type=str,
    # )

    return str(
        {
            "board_fen": fen,
            "proposed_move": move,
            "is_check": is_check,
            "stockfish_evaluation": evaluation,
            "best_continuation": continuation,
            "comment": comment,
        }
    )
