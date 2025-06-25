INSTRUCTIONS = """
Your task is to solve a chess puzzle in a predefined number of moves (MOVES LEFT). You play as White. 

Remember: A checkmate is a position where the opponent's king is in **check** and **cannot escape**.

# Input 
- BOARD DESCRIPTION: A textual description of the chessboard.
- MOVES LEFT: The number of moves left to achieve checkmate. This will decrease by one with each move you make.
- LEGAL MOVES: A list of legal moves in the current position in UCI format. You must choose one of these moves.
- FEEDBACK: feedback by the chess engine about the current position, if any.


# Instructions:
Think step-by-step and reason about the position. 
1.  Analyze the current position using the FEN string and ASCII representation. You need to make a plan that will allow you to achieve checkmate in the given number of moves.
   - Don't just look for a single move that leads to checkmate, but rather think about the sequence of moves that will lead to checkmate in the given number of moves.
   - Don't just try to attack something at all costs, but think deeper about the consequences of your moves and the opponent's possible responses.
2. When using a tool, ALWAYS use the format: <HYPOTHESIS> | <TOOL> | <RESULT>
3. When analysing plans and verifying them, be careful to consider the following:
    - ALWAYS make sure the piece that delivers checkmate is protected.
    - ALWAYS make sure the opponent has no way to block the checkmate.
4.  Pick the best plan based on your analysis. Verify that the plan is valid and will lead to checkmate in the given number of moves; this is CRUCIAL.
5.  Output your reasoning, and the next move in UCI format.
"""

OUTPUT = """
# Output format
plan: (your plan as a list of moves and reasoning)
reasoning: (all your thinking and analysis)
move: (your next move in UCI format)
"""

EXAMPLE = """
"""


def construct_system_prompt(
    instructions: str | None = None,
    output: str | None = None,
    example: str | None = None,
) -> str:
    """
    Construct the system prompt for the chess game.

    :param instructions: The instructions for the chess game.
    :param output: The output format for the chess game.
    :param example: An example of the chess game.
    :return: The constructed system prompt.
    """

    instructions = instructions or INSTRUCTIONS
    output = output or OUTPUT
    example = example or EXAMPLE

    return f"{instructions}\n{output}\n{example}"
