# prompts for the chess game

INSTRUCTIONS = """
Your task is to play strong chess to solve a chess puzzle in a predefined number of moves (MOVES LEFT). You play as White. 

# Input 
- FEN: The current position of the chess game in FEN format.
- MOVES LEFT: The number of moves left to achieve checkmate. This will decrease by one with each move you make.
- LEGAL MOVES: A list of legal moves in the current position in UCI format. You must choose one of these moves.
- FEEDBACK: feedback by the chess engine about the current position, if any.

# Instructions:
1.  Analyze the current position using the FEN string. Look for forcing moves that you can use as plans to achieve checkmate in the given number of moves.
    - A plan is a sequence of moves that leads to checkmate.
    - Checkmate is a position where the opponent's king is in check and has no legal moves to escape (you need both conditions).
    - Come up with ideas for how to achieve checkmate in the given number of moves.
    - Use the tools available to verify your ideas and plans. It's especially important to verify that you have the right number of moves left to achieve checkmate.
2. When you use a tool, state what is the hypothesis you are testing, which tool you are using, and what is the result of the tool.
- Common things to check:
    - Your move sacrifices a piece but gets no advantage or makes the position worse.
    - Your move is not a checkmate in the given number of moves.
    - Your move does not lead to a winning position.
3.  Pick the best plan based on your analysis and reasoning and execute the first move of that plan (making sure it is a legal move).
4.  Output your reasoning, and the next move in UCI format.
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
