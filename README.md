# Chess-Puzzle-Loving agent

This implements a little LLM agent that tries to solve chess puzzles against an opponent that makes random moves.

For every turn, first a random move is picked, then scored against Stockfish and its best counter move. Then:

1. the agent criticizes the random move in the context of what stockfish did and then
2. proposes a move that should be better.

and this 1-2 loop continues up to some max number of tries (default is `3`) or until we find a good move (whatever happens first). The best move per turn (according to Stockfish score) is used.

This strategy can discover some nice solutions, but doesn't always work.

## Installation

```bash
uv pip install . 
```

## Usage 

The main script is 
