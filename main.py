import dspy
from tqdm import tqdm
import os
import chess
import random
from dotenv import load_dotenv
import concurrent.futures
from fire import Fire
import chess.svg


# Load environment variables from .env file
assert load_dotenv()


class PlayerA(dspy.Signature):
    """You are an expert chess analysis engine. Your purpose is to identify the optimal move in a given chess position, operating under specific tactical constraints.

    # Primary Objective
    Find a sequence of moves that leads to a forced checkmate within the `number_of_remaining_moves`.

    ---

    # Context and Inputs
    You will be provided with the following information for each turn:

    - **`player_color`**: The color you are playing (e.g., 'white').
    - **`board_fen`**: The current state of the chessboard in Forsyth-Edwards Notation (FEN).
    - **`legal_moves`**: A list of all legal moves available to you in the current position (in UCI format).
    - **`number_of_remaining_moves`**: The maximum number of moves you have left to achieve the Primary Objective. If this is 1, you must deliver checkmate on this move.
    - **`additional_instructions`**: Optional strategic guidance to consider (e.g., "Prioritize saving the h-pawn" or "Attempt to trade queens").

    ---

    # Reasoning Process
    Before providing your final move, you must reason through the position by following these steps:

    1.  **Board Assessment**: Briefly analyze the current board state (`board_fen`). Identify immediate threats from your opponent, your own tactical opportunities, material balance, and key positional features.
    2.  **Objective Check**: Determine if the `Primary Objective` is achievable. Search for candidate moves that could lead to a forced checkmate within the `number_of_remaining_moves`.
        * If you find one or more mating sequences, outline the most efficient one.
    3.  **Fallback Plan**: If no forced mate within the limit is found, activate the `Secondary Objective`. Evaluate the top candidate moves based on their potential to improve your position. Consider factors like:
        * Forcing a future checkmate sequence.
        * Gaining a significant material advantage.
        * Seizing control of the center, improving piece activity, or creating weaknesses in the opponent's structure.
    4.  **Threat Verification**: For your chosen move, perform a final check for any immediate blunders or missed threats from the opponent. Ensure your move is safe and advances your plan.
    5.  **Plan Formulation**: State your chosen move and provide a concise one-sentence summary of your plan.

    ---

    # Output Format
    Your final response must be the single best move you have identified, expressed in Universal Chess Interface (UCI) format.

    # Example Output:
    e2e4
    ```
    """

    fen: str = dspy.InputField(
        description="The FEN string representing the current state of the chessboard."
    )
    legal_moves: list = dspy.InputField(
        description="A list of legal moves in the current position, represented in UCI format (e.g., 'e2e4')."
    )
    number_of_remaining_moves: int = dspy.InputField(
        description="The number of moves left in the game.",
    )
    additional_instructions: str = dspy.InputField(
        description="Additional instructions to take into account when considering the best move to make.",
        default="",
    )
    move: str = dspy.OutputField(
        description="The best move to make in the current position, in UCI format (e.g., 'e2e4').",
    )


class PlayerB(dspy.Signature):
    """
    Play grandmaster-level chess against an opponent. Your objective is to win the game by making the best moves possible within the
    number of moves left in the game.

    # Instructions
    - You have a limited number of moves to make, you must win within the number of moves left in the game. If number_of_remaining_moves is 1, you must checkmate your opponent the next move.
    - Reason about the best move to make in the current position, considering:
        1. the legal moves available
        2. the number of moves left in the game,
        3. any additional_instructions.
        4. and, most importantly, the plan from the previous move. Review if it still makes sense, and if it does, use it, otherwise revise it. Infer what black did based on what you see on the board as well as your plan.
    - Make a plan about how you will win the game in the number_of_remaining_moves. Consider the opponent's threats and how to counter them.
    - Respond with your next move in UCI format (e.g., 'e2e4').
    - Also respond with your short plan for the next move. It should be formatted as a list of bullet points, with each bullet point starting with a dash (-) and a space. Mention what you did in the previous move, and what you plan to do in the next move.
    """

    fen: str = dspy.InputField(
        description="The FEN string representing the current state of the chessboard."
    )
    legal_moves: list = dspy.InputField(
        description="A list of legal moves in the current position, represented in UCI format (e.g., 'e2e4')."
    )
    number_of_remaining_moves: int = dspy.InputField(
        description="The number of moves left in the game.",
    )
    additional_instructions: str = dspy.InputField(
        description="Additional instructions to take into account when considering the best move to make.",
        default="",
    )
    plan_from_previous_move: str = dspy.InputField(
        description="A short plan from the previous move, formatted as a list of bullet points.",
        default="",
    )
    move: str = dspy.OutputField(
        description="The best move to make in the current position, in UCI format (e.g., 'e2e4').",
    )
    plan: str = dspy.OutputField(
        description="A short plan for the next move, formatted as a list of bullet points.",
        default="",
    )


def random_player(board) -> str:
    """
    Given a board, get all of the legal moves in the current position.
    Then pick one of the legal moves at random and return it in UCI format.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        assert False, "No legal moves available, cannot play."
    move = random.choice(legal_moves)
    return move.uci()


def play_game(
    fen: str,
    llm_player,
    move_limit: int = 2,
    use_valid_move_checker: bool = False,
    use_plan: bool = False,
) -> str:
    """
    Play a game of chess against a random player.
    """

    board = chess.Board(fen)
    plan = None
    while not board.is_game_over():
        if move_limit <= 0:
            print("Move limit reached, ending game.")
            break

        legal_moves = [m.uci() for m in board.legal_moves]
        move = llm_player_move(
            llm_player, move_limit, board, legal_moves, use_valid_move_checker, plan
        )
        plan = move.plan.strip() if use_plan else None
        board.push_uci(move.move)
        move_limit -= 1

        print("reasoning:", move.reasoning)

        if board.is_game_over():
            break

        # random player makes a move
        random_move = random_player(board)
        board.push_uci(random_move)

        print(board.unicode())

    # print the result for white and black
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            print("Black wins by checkmate.")
        else:
            print("White wins by checkmate.")

    return board.result()


def llm_player_move(
    llm_player,
    move_limit,
    board,
    legal_moves,
    use_valid_move_checker: bool = False,
    plan: str | None = None,
):
    additional_instructions = ""

    for _ in range(3):
        if plan:
            move = llm_player(
                fen=board.fen(),
                legal_moves=legal_moves,
                number_of_remaining_moves=move_limit,
                additional_instructions=additional_instructions,
                plan_from_previous_move=plan,
            )
            plan = move.plan.strip()
        else:
            move = llm_player(
                fen=board.fen(),
                legal_moves=legal_moves,
                number_of_remaining_moves=move_limit,
                additional_instructions=additional_instructions,
            )

        llm_move = move.move.strip()

        if use_valid_move_checker and (not llm_move or llm_move not in legal_moves):
            additional_instructions += f"- The move you proposed: {llm_move} is not a legal move. Please check the legal moves and try again.\n"
            print(f"LLM move {llm_move} is not legal, retrying.")
            continue

        if use_valid_move_checker and move_limit == 1:
            temp_board = board.copy()
            temp_board.push_uci(llm_move)
            if not temp_board.is_checkmate():
                print(f"LLM move {llm_move} is not a checkmate or stalemate, retrying.")
                additional_instructions += f"- You proposed the move: {llm_move}, however a chess-verifier checked it and it is not a checkmate. Please pick a new move and be careful of hallucinations.\n"
            else:
                break
        else:
            break

    if (llm_move not in legal_moves) or (not llm_move):
        # play randomly
        print("No move provided by LLM, playing randomly.")
        move.move = random_player(board)
    return move


def main(
    number_of_trials: int = 5,
    parallel_games: int = 1,
    use_valid_move_checker: bool = False,
    use_plan: bool = False,
) -> None:
    """
    Load the first puzzle from a file and play the puzzle number_of_trials times in parallel.
    Then output statistics about the results of the game.
    """
    lm = dspy.LM(
        model="openrouter/anthropic/claude-3-7-sonnet-20250219",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        cache=False,
    )

    dspy.configure(lm=lm)

    if use_plan:
        llm_player = dspy.ChainOfThought(PlayerB)
    else:
        llm_player = dspy.ChainOfThought(PlayerA)

    # from Siegbert Tarrasch vs. Max Kurschner, mate in 2
    # https://www.sparkchess.com/chess-puzzles/siegbert-tarrash-vs-max-kurschner.html
    move_limit, puzzle = (
        2,
        "r2qk2r/pb4pp/1n2Pb2/2B2Q2/p1p5/2P5/2B2PPP/RN2R1K1 w - - 1 0",
    )

    # mate in 3
    # https://www.sparkchess.com/chess-puzzles/dawid-przepiorka-vs-erich-eliskases.html
    # move_limit, puzzle =3,  2r3k1/p4p2/3Rp2p/1p2P1pK/8/1P4P1/P3Q2P/1q6 b - - 0 1

    if puzzle:
        if chess.Board(puzzle).turn == chess.BLACK:
            print("Puzzle is for black, flipping to white.")
            puzzle = chess.Board(puzzle).transform(chess.flip_vertical).fen()

    fen = puzzle.strip()
    count_wins = 0
    count_draws = 0

    def run_game(_):
        return play_game(
            fen,
            llm_player,
            move_limit,
            use_valid_move_checker=use_valid_move_checker,
            use_plan=use_plan,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_games) as executor:
        results = list(
            tqdm(
                executor.map(run_game, range(number_of_trials)), total=number_of_trials
            )
        )

    for result in results:
        print(f"Result for puzzle {fen}: {result}")
        if result == "1-0":
            count_wins += 1
        elif result == "1/2-1/2":
            count_draws += 1

    print(f"\nResults for puzzle {fen}: {count_wins} wins, {count_draws} draws")


if __name__ == "__main__":
    Fire(main)
