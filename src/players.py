import asyncio
import random
import chess
from pydantic import BaseModel, Field
from pydantic_ai import Agent
import chess.engine
from pydantic_ai.models.openai import OpenAIModel
from src.prompts import INSTRUCTIONS_COACH
from dotenv import load_dotenv

load_dotenv()

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

model_name = "anthropic/claude-3-7-sonnet-20250219"
lm = OpenAIModel(
    model_name=model_name,
    provider="openrouter",
)


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


class FeedbackModel(BaseModel):
    """
    Model for feedback on the player's move.
    This model can be extended to include additional fields as needed.
    """

    reasoning: str = Field(
        ..., description="reasoning behind the feedback on the player's move."
    )
    feedback: str = Field(
        ...,
        description="One sentence feedback on the player's move, including suggestions for improvement. The feedback should be concise and actionable and include the score. Also mentioning the player move and the best counter move by the opponent.",
    )


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
            move_is_legal,
            # get_attackers_of_squares,
            # are_sequences_checkmate,
            # get_best_legal_counter_move_by_opponent,
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
        max_tries: int = 5,
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
        moves_and_scores = []
        while tries < max_tries:
            input_prompt = self.create_instructions_str(
                board,
                legal_moves,
                move_limit,
                ((additional_instructions or "") + " " + (feedback or "")).strip(),
            )

            move = asyncio.run(
                self.run(
                    input_prompt,
                    model_settings=model_settings,
                )
            ).output

            print(f"Model generated move: {move.move} with reasoning: {move.reasoning}")
            print(
                f"Model plan: {move.plan if hasattr(move, 'plan') else 'No plan provided'}"
            )

            if not move:
                print("No move generated by the model, retrying.")
                tries += 1
                continue

            if checkmate_retry:
                # check if move has an attribute plan
                new_feedback, score = self.check_if_plan_is_valid(
                    move.plan[0:1], board, move_limit
                )
                moves_and_scores.append((move, score))
                if new_feedback:
                    tries += 1
                    print(f"new_feedback.feedback: {new_feedback.feedback}")
                    feedback += f"feedback[{tries}] : {new_feedback.feedback}\n"
                    continue
                else:
                    break

        # return move with best score

        return (
            max(moves_and_scores, key=lambda x: x[1])[0] if moves_and_scores else None
        ), moves_and_scores

    def check_if_plan_is_valid(
        self, plan: list[str], board: chess.Board, move_limit: int
    ) -> tuple[FeedbackModel, float]:
        """
        Check if the proposed plan is valid by simulating the moves on the board. A plan is valid if
        it leads to checkmate in the specified number of moves.

        :param plan: A list of moves in UCI format (e.g., ['e2e4', 'e7e5']). Those are only the moves that the model has proposed.
        :param board: A chess.Board object representing the current position.
        :param move_limit: The number of moves left in the game.
        :return: None if the plan is valid, or a string with feedback if the plan is not valid.
        """
        # for each move in the plan, simulate the move on the board
        temp_board = board.copy()

        tactical_agent = Agent(
            model=lm, system_prompt=INSTRUCTIONS_COACH, output_type=FeedbackModel
        )

        record_best_engine_moves = []
        legal_moves = [m.uci() for m in temp_board.legal_moves]

        if plan[0] not in legal_moves:
            return (
                FeedbackModel(
                    reasoning="The first move in the plan is not legal.",
                    feedback=f"The first move in the plan {plan[0]} is not valid. Please check your plan and ensure all moves are legal.",
                ),
                -10000.0,
            )

        if move_limit == 1:
            temp_board.push_uci(plan[0])
            if temp_board.is_checkmate():
                return None, 10000

            temp_board.pop()

        # score each move in the plan against Stockfish
        scores = {}
        for move in plan:
            score = score_move_against_stockfish(move, temp_board.fen())
            scores[move] = score

        print(f"Scores for the plan: {scores}")

        for i, move in enumerate(plan):
            if scores[move] > 10:
                # move is good, review the next
                continue
            try:
                legal_moves = [m.uci() for m in temp_board.legal_moves]

                if move not in legal_moves:
                    feedback = asyncio.run(
                        tactical_agent.run(
                            str(
                                {
                                    "board_fen": board.fen(),
                                    "plan with chess engine scores": scores,
                                    "opponents_counter_moves": record_best_engine_moves,
                                    "move_limit": move_limit,
                                    "move to review": move,
                                }
                            )
                        )
                    ).output
                    print(f"Feedback from tactical agent: {feedback}")
                    return feedback, scores[move]

                move_obj = chess.Move.from_uci(move)
                temp_board.push(move_obj)

                # If it's the last move in the plan, check for checkmate
                if i == len(plan) - 1:
                    if not temp_board.is_checkmate():
                        feedback = asyncio.run(
                            tactical_agent.run(
                                str(
                                    {
                                        "board_fen": board.fen(),
                                        "plan with chess engine scores": scores,
                                        "opponents_counter_moves": record_best_engine_moves,
                                        "move_limit": move_limit,
                                        "move to review": move,
                                    }
                                )
                            )
                        ).output
                        print(f"Feedback from tactical agent: {feedback}")
                        return feedback, scores[move]

                # If it's not the last move, get the best response from the opponent
                if i < len(plan) - 1:
                    counter_move = get_best_legal_counter_move_by_opponent(temp_board)
                    if counter_move:
                        counter_move_obj = chess.Move.from_uci(counter_move)
                        temp_board.push(counter_move_obj)
                        record_best_engine_moves.append(counter_move)

            except ValueError as e:
                return f"Error processing move {move}: {str(e)}", 0.0

        return None, 10000

    def create_instructions_str(
        self,
        board: chess.Board,
        legal_moves: list[str],
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
        print(f"len(legal_moves): {len(legal_moves)}")
        prompt = f"""
        - BOARD DESCRIPTION: {board.fen()}
        - LEGAL MOVES: {", ".join(legal_moves)}
        - n: {move_limit}
        """

        if additional_instructions:
            prompt += f" - FEEDBACK: {additional_instructions}"

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
