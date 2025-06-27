INSTRUCTIONS = """
Your task is to solve a chess puzzle in a predefined number of moves (MOVES LEFT). You play as White. 

Remember: A checkmate is a position where the opponent's king is in **check** and **cannot escape**.

# Input 
- BOARD DESCRIPTION: A textual description of the chessboard.
- MOVES LEFT: The number of moves left to achieve checkmate. This will decrease by one with each move you make.
- LEGAL MOVES: A list of legal moves in the current position in UCI format. You must choose one of these moves.
- FEEDBACK: feedback by the chess engine about the current position, if any.

# Useful things to remember: 
- You can only promote a pawn to a queen, rook, bishop, or knight when you reach the last rank, which for white is the 8th rank.
- A piece cannot pass through a square occupied by your own piece or the opponent's piece, except for the knight, which can jump over pieces.

# Instructions:
Think deeply and step-by-step about the position. Use the following steps to construct your plan: 
1.  Analyze the current position using the FEN string and ASCII representation. You need to make a plan that will allow you to achieve checkmate in the given number of moves.
    - If MOVES LEFT is 1, you need to find a checkmate in one move. Check if there's a direct checkmate available.
    - If MOVES LEFT is greater than 1, think about how to create a position where you can checkmate the opponent's king in the given number of moves.
    - Look for tactical opportunities, such as forks, pins, and skewers.
   - Don't rush to capture, think about the opponent's threats and how to neutralize them.
   - When constructing plans, ALWAYS consider the best response from the opponent.
2. When using a tool, ALWAYS use the format: <HYPOTHESIS> | <TOOL> | <RESULT>
3.  Pick the best plan based on your analysis. Verify that the plan is valid and will lead to checkmate in the given number of moves; this is CRUCIAL.
4.  Output your reasoning, and the next move in UCI format.
"""

ENHANCED_INSTRUCTIONS = """
you are a chess agent and you will help me solve a mate-in-two puzzle.

Review the FEN, you represent the white player. Then think step-by-step about strategies and give me a list of your two moves in UCI format.

You may use the tools to verify your hypotheses about the position and check if your plan is valid.

# Input 
- BOARD DESCRIPTION: A textual description of the chessboard.
- MOVES LEFT: The number of moves left to achieve checkmate. This will decrease by one with each move you make.
- LEGAL MOVES: A list of legal moves in the current position in UCI format. You must choose one of these moves.
- FEEDBACK: feedback by the chess engine about the current position, if any.

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
    use_enhanced: bool = False,
) -> str:
    """
    Construct the system prompt for the chess game.

    :param instructions: The instructions for the chess game.
    :param output: The output format for the chess game.
    :param example: An example of the chess game.
    :param use_enhanced: Whether to use enhanced instructions with tactical patterns.
    :return: The constructed system prompt.
    """

    if instructions is None:
        instructions = ENHANCED_INSTRUCTIONS if use_enhanced else INSTRUCTIONS
    output = output or OUTPUT
    example = example or EXAMPLE

    return f"{instructions}\n{output}\n{example}"
