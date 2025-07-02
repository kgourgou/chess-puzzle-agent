# Chess-Puzzle-Loving agent

This implements a little LLM agent that tries to solve chess puzzles against an opponent that makes random moves.

For every turn, first a random move is picked, then scored against Stockfish and its best counter move. Then:

```markdown
for _ in range(max_tries): 
    1. The LLM is prompted with the current board state, the best counter move, and the score of the best counter move.
    2. The LLM suggests a move.
    3. If the move is legal, it is scored against Stockfish.
    4. If the score is better than the previous best score, it becomes the new best move.
```

This strategy can discover some nice solutions, but doesn't always work.

## Installation & Requirements

First, you will need [stockfish](https://stockfishchess.org/) and a connection to an LLM (I mostly used OpenRouter, but you should be able to use any model pydantic-ai supports).

The env file should look like this:

```bash
OPENROUTER_API_KEY = <your-api>
OPENROUTER_API_URL = https://openrouter.ai/api/v1

LOGFIRE_KEY = <logfire-key> # optional, for logging
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

MODEL_NAME = "anthropic/claude-3-7-sonnet-20250219"
```

Then, install the requirements:

```bash
uv pip install . 
```

## Usage

Check `main.py`, you can run it as `uv run python main.py --mate_in_k 2`.

Also, Claude Code wrote a nice ui if you want to play against the agent that way:

```bash
uv run python web_ui.py
```

### RandomCheckPlayer

The repository includes a `RandomCheckPlayer` that tests the hypothesis that randomly selecting good-scoring check moves might be sufficient for puzzle solving. This player:

- Finds all legal moves that deliver check
- Randomly selects from these check moves
- Scores each move with Stockfish
- Uses moves that exceed a configurable score threshold
- Falls back to the best-scoring check move if no move exceeds the threshold

### Experiment Script

Run `test_random_check_hypothesis.py` to compare the RandomCheckPlayer against the LLM player:

```bash
uv run python test_random_check_hypothesis.py --mate_in_k 2 --number_of_trials 20
```

This experiment script:

- Runs both players on the same puzzles
- Collects detailed performance metrics (win rates, move quality, efficiency)
- Performs statistical analysis comparing the approaches
- Generates move-by-move SVG visualizations (optional)
- Tests whether simple check-based heuristics can match LLM performance
