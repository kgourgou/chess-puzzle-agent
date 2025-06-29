from tqdm import tqdm
import chess
from dotenv import load_dotenv
import concurrent.futures
from fire import Fire
import mlflow


import os
import chess.svg
from pydantic import Field
from pydantic_ai.models.openai import OpenAIModel

from dataclasses import dataclass

from src.players import (
    random_player,
    LLMPlayer,
    ChessBM,
    CorrectorLLMPlayer,
    FeedbackModelWithMove,
)
from src.prompts import INSTRUCTIONS_COACH


# Load environment variables from .env file
assert load_dotenv()

# configure logfire
if "LOGFIRE_KEY" in os.environ:
    import logfire

    logfire.configure(token=os.getenv("LOGFIRE_KEY"))
    logfire.instrument_pydantic_ai()


@dataclass
class PlayGameConfig:
    move_limit: int = 2
    use_plan: bool = False
    show_board: bool = True
    show_reasoning: bool = True
    checkmate_retry: bool = False
    svg: bool = False


def play_game(
    mlflow_run_id: str,
    board: chess.Board,
    llm_player: LLMPlayer,
    config: PlayGameConfig,
) -> str:
    """
    Play a game of chess against a random player.

    :param mlflow_run_id: The MLflow run ID to log the game artifacts.
    :param board: A chess.Board object representing the current position.
    :param llm_player: An instance of LLMPlayer that will make moves.
    :return: The result of the game as a string (e.g., "1-0", "0-1", or "1/2-1/2").
    """

    # Ensure a fresh board instance for thread safety
    board = chess.Board(board.fen())

    if config.show_board:
        print(board.unicode())

    move_limit = config.move_limit

    # --- SVG saving setup ---
    svg_dir = None
    if config.svg:
        svg_dir = os.path.join("svgs", mlflow_run_id)
        os.makedirs(svg_dir, exist_ok=True)
        move_idx = 0
        # Save initial position
        svg_path = os.path.join(svg_dir, f"move_{move_idx:03d}.svg")
        with open(svg_path, "w") as f:
            f.write(chess.svg.board(board=board))
        move_idx += 1
    # --- end SVG setup ---

    while not board.is_game_over():
        if move_limit <= 0:
            print("Move limit reached, ending game.")
            break

        move, moves_and_scores = llm_player_move(
            llm_player=llm_player,
            move_limit=move_limit,
            board=board,
            checkmate_retry=config.checkmate_retry,
        )

        # plt.plot(
        #     [i for i in range(len(moves_and_scores))],
        #     [score for _, score in moves_and_scores],
        #     marker="o",
        # )
        # plt.xlabel("Try")
        # plt.ylabel("Score")
        # plt.title("Scores of Moves Over Tries")
        # plt.show()
        # exit(0)

        board.push_uci(move.move)
        move_limit -= 1

        if config.show_reasoning:
            print("reasoning:", move.reasoning)
            print("LLM player move:", move.move)

        if config.show_board:
            print(board.unicode())

        # --- SVG: save after LLM move ---
        if config.svg:
            svg_path = os.path.join(svg_dir, f"move_{move_idx:03d}.svg")
            with open(svg_path, "w") as f:
                f.write(chess.svg.board(board=board))
            move_idx += 1

        if board.is_game_over():
            break

        # random player makes a move
        random_move = random_player(board)
        board.push_uci(random_move)
        if config.show_reasoning:
            print(f"Random player move: {random_move}")

        if config.show_board:
            print(board.unicode())

        # --- SVG: save after random move ---
        if config.svg:
            svg_path = os.path.join(svg_dir, f"move_{move_idx:03d}.svg")
            with open(svg_path, "w") as f:
                f.write(chess.svg.board(board=board))
            move_idx += 1

    # print the result for white and black
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            print("Black wins by checkmate.")
        else:
            print("White wins by checkmate.")

    if config.svg and svg_dir:
        print(f"Logging SVGs to MLflow at {svg_dir}")
        mlflow.log_artifacts(svg_dir)

    return board.result()


def llm_player_move(
    llm_player,
    move_limit,
    board,
    checkmate_retry: bool = False,
    temperature: float = 0.2,
):
    move = llm_player.move(
        board,
        move_limit,
        checkmate_retry=checkmate_retry,
        model_settings={"temperature": temperature},
    )

    return move


def main(
    number_of_trials: int = 1,
    mate_in_k: int = 2,
    parallel_games: int = 1,
    use_plan: bool = True,
    show_board: bool = False,
    show_reasoning: bool = False,
    use_example: bool = False,
    checkmate_retry: bool = True,
    svg: bool = False,
    use_enhanced_prompts: bool = True,
) -> None:
    """

    Main function to run the chess game simulation with the specified parameters.
    :param number_of_trials: Number of trials to run.
    :param parallel_games: Number of games to run in parallel.
    :param use_plan: If True, the LLM will generate a plan for the game.
    :param show_board: If True, the board will be displayed after each move.
    :param show_reasoning: If True, the reasoning behind the LLM's move will be displayed.
    :param use_example: If True, an example will be used in the prompt.
    :param checkmate_retry: If True, the LLM will retry if the move does not result in checkmate.
    :param use_enhanced_prompts: If True, use enhanced prompts with tactical patterns and better guidance.

    """
    mlflow.set_experiment("chess_game_simulation")
    mlflow.set_tracking_uri("mlruns")
    mlflow.openai.autolog()

    model_name = os.getenv("MODEL_NAME")
    lm = OpenAIModel(
        model_name=model_name,
        provider="openrouter",
    )

    example = None
    output = prep_output(use_plan)
    # chess_bm = ChessBM

    if use_plan:
        print("Using plan in the output.")

        class ChessBMP(ChessBM):
            plan: list[str] = Field(
                ...,
                description="List of moves in UCI format with reasoning for each move.",
            )

        # chess_bm = ChessBMP

    #instructions_prompt = construct_system_prompt(
    #    output=output, example=example, use_enhanced=use_enhanced_prompts
    #)

#    print("Using instructions_prompt:")
 #   print(instructions_prompt)

    # llm_player = LLMPlayer(
    #     model=lm,
    #     instructions=instructions_prompt,
    #     output_type=chess_bm,
    #     name="Player A",
    #     retries=3,
    # )
    print(f'prompt: {INSTRUCTIONS_COACH}')
    llm_player = CorrectorLLMPlayer(
        model=lm,
        output_type=FeedbackModelWithMove,
        instructions=INSTRUCTIONS_COACH,
        name="Player A",
        retries=3,
    )

    move_limit, fen, board = prep_puzzle(mate_in_k=mate_in_k)

    config = PlayGameConfig(
        move_limit=move_limit,
        use_plan=use_plan,
        show_board=show_board,
        checkmate_retry=checkmate_retry,
        svg=svg,
    )

    # don't save SVGs if parallel games > 1
    config.svg = svg if parallel_games == 1 else False

    if svg:
        print("Saving game as a GIF. SVGs will be saved in mlflow artifacts.")

    count_wins, count_draws, count_losses = 0, 0, 0

    with mlflow.start_run():
        mlflow_run_id = mlflow.active_run().info.run_id

        def run_game():
            return play_game(mlflow_run_id, board, llm_player, config)

        if parallel_games == 1:
            # run games sequentially
            results = []
            for i in tqdm(range(number_of_trials), desc="Running games"):
                result = run_game()
                results.append(result)
        else:
            # run games in parallel
            print(
                f"Running {number_of_trials} games in parallel with {parallel_games} workers."
            )
            # Use ThreadPoolExecutor for parallel execution
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=parallel_games
            ) as executor:
                results = list(
                    tqdm(
                        executor.map(run_game, range(number_of_trials)),
                        total=number_of_trials,
                    )
                )

        count_wins, count_draws, count_losses = count_puzzle_outcomes(fen, results)

        print(
            f"\nTotal results for puzzle {fen}: {count_wins} wins, {count_draws} draws, {count_losses} losses"
        )

        # log the results into MLflow
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("puzzle_fen", fen)
        mlflow.log_param("number_of_trials", number_of_trials)
        mlflow.log_param("parallel_games", parallel_games)
        mlflow.log_param("use_plan", use_plan)
        mlflow.log_param("show_board", show_board)
        mlflow.log_param("show_reasoning", show_reasoning)
        mlflow.log_param("move_limit", config.move_limit)
        mlflow.log_param("use_example", use_example)
        mlflow.log_param("use_enhanced_prompts", use_enhanced_prompts)
        mlflow.log_metric("losses", count_losses)
        mlflow.log_metric("wins", count_wins)
        mlflow.log_metric("draws", count_draws)


def count_puzzle_outcomes(fen, results):
    count_wins, count_draws, count_losses = 0, 0, 0
    for result in results:
        print(f"Result for puzzle {fen}: {result}")
        if result == "1-0":
            count_wins += 1
        elif result == "1/2-1/2":
            count_draws += 1
        elif result == "0-1":
            count_losses += 1
    return count_wins, count_draws, count_losses


def prep_puzzle(mate_in_k=2):
    """
    Prepare a chess puzzle based on the mate in k moves.
    :param mate_in_k: The number of moves to mate in (2 or 3).
    :return: A tuple containing the move limit, FEN string, and chess board.
    """
    puzzle = None
    if mate_in_k == 1:
        move_limit, puzzle = (1, "7r/p2B1ppp/k2p2b1/4n3/1Q2P3/P1B5/KP3P1P/7q w - - 0 3")

    elif mate_in_k == 2:
        # from Siegbert Tarrasch vs. Max Kurschner, mate in 2
        # https://www.sparkchess.com/chess-puzzles/siegbert-tarrash-vs-max-kurschner.html
        # move_limit, puzzle = (
        #     2,
        #     "r2qk2r/pb4pp/1n2Pb2/2B2Q2/p1p5/2P5/2B2PPP/RN2R1K1 w - - 1 0",
        # )

        # https://www.sparkchess.com/chess-puzzles/paul-morphys-problem.html
        move_limit, puzzle = (2, "kbK5/pp6/1P6/8/8/8/8/R7 w - -")

        # move_limit, puzzle = (
        #     2,
        #     "r4R2/1b2n1pp/p2Np1k1/1pn5/4pP1P/8/PPP1B1P1/2K4R w - - 1 0",
        # )
    elif mate_in_k == 3:
        # https://www.sparkchess.com/chess-puzzles/roberto-grau-vs-edgar-colle.html
#        move_limit, puzzle = (
#            3,
#            "1k5r/pP3ppp/3p2b1/1BN1n3/1Q2P3/P1B5/KP3P1P/7q w - - 1 0",
#        )
        move_limit, puzzle = (3, "R6R/1r3pp1/4p1kp/3pP3/1r2qPP1/7P/1P1Q3K/8 w - - 1 0")
    else:
        raise ValueError(f"k = {mate_in_k} is not supported.")

    if puzzle:
        if chess.Board(puzzle).turn == chess.BLACK:
            print("Puzzle is for black, flipping to white.")
            puzzle = chess.Board(puzzle).transform(chess.flip_vertical).fen()

        fen = puzzle.strip()
        board = chess.Board(fen)
        if not board.is_valid():
            raise ValueError(f"Invalid FEN: {fen}")
    return move_limit, fen, board


def prep_output(use_plan):
    output_no_plan = """# Output format\n reasoning: (all your thinking and analysis)\n move: (your next move in UCI format)"""
    output_with_plan = """# Output format\n plan: <your-plan-as-a-list-of-n-moves>\n reasoning: (all your thinking and analysis)\n move: (your next move in UCI format)
    
Example output: 
reasoning: <some-long-reasoning-about-the-position-and-your-plan, mentions the moves Nf3 and d4 as moves for white>
move: Nf3
plan: [Nf3, d4]
    """

    output = output_with_plan if use_plan else output_no_plan
    return output


if __name__ == "__main__":
    Fire(main)
