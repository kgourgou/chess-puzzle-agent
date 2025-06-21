# prompts for the chess game

INSTRUCTIONS = """
Your task is to play strong chess to solve a chess puzzle in predefined number of moves. You play as white and your opponent plays as black. You will need at least two moves to achieve checkmate, so don't try to solve the puzzle in one move.


# Input 
- FEN: The current position of the chess game in FEN format.
- MOVES LEFT: The number of moves left to achieve checkmate.
- LEGAL MOVES: A list of legal moves in the current position in UCI format.
- FEEDBACK: feedback by the chess engine about the current position, if any.

# Instructions:
0. **Review the feedback** if provided. It may contain hints or corrections to your previous move.
1.  **State the Goal:** Begin by explicitly stating the objective. For example, "My goal is to find a forced checkmate in N moves."
2.  **Candidate Move Generation:** Identify a few promising candidate moves for your first move (let's call it W1). Prioritize them using this hierarchy:
    a.  **Checks:** Forcing moves that restrict Black's king.
    b.  **Captures:** Moves that remove a key defender or open lines.
3.  **Recursive Verification:** For your top candidate move (W1), you must prove it leads to a win. To do this, simulate the following logic in your reasoning:
    a.  "If I play W1, what are Black's best replies (B1)?"
    b.  "For **every** reply B1, I must find a response (W2) that leads to a forced mate in N-1 moves."
    c.  "To prove the line for W2, I must consider all of Black's subsequent replies (B2) and show that for each one, I can force a mate in N-2 moves."
    d.  Continue this thought process until you can show a final, undeniable checkmating move.
4.  **Construct a Plan:** If your analysis shows that your candidate move W1 forces a checkmate against **all** of Black's defenses within N moves, you have found the solution. Summarize the main lines of play. If any single defense by Black allows them to escape, your candidate move W1 is wrong and you must discard it and analyze your next candidate.
5.  **Choose your next move** from the legal moves.
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
