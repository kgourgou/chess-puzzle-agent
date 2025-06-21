import requests
import chess
import chess.pgn
import io
from functools import lru_cache


@lru_cache(maxsize=3)
def get_daily_puzzle() -> dict:
    try:
        # URL for a single random Lichess puzzle
        url = "https://lichess.org/api/puzzle/daily"

        response = requests.get(url)
        response.raise_for_status()
        puzzle_data = response.json()

        return puzzle_data

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def convert_to_fen(puzzle: dict) -> str:
    pgn = puzzle["game"]["pgn"]
    pgn_io = io.StringIO(pgn)
    game = chess.pgn.read_game(pgn_io)
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)

    return board.fen()


if __name__ == "__main__":
    puzzle = get_daily_puzzle()
    if puzzle:
        print("Daily puzzle retrieved successfully.")
        fen = convert_to_fen(puzzle)
        print(f"Daily Puzzle FEN: {fen}")

        with open("daily_puzzle.csv", "w") as f:
            f.write(f"{fen}\n")
        print("Daily puzzle saved to daily_puzzle.csv")

    else:
        print("Failed to retrieve the daily puzzle.")
