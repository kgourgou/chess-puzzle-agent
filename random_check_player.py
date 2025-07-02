import random
import chess
from typing import List, Optional
from src.tools import score_move_against_stockfish, is_check
from src.players import ChessBM


class RandomCheckPlayer:
    """
    A chess player that selects moves randomly from all legal moves that deliver check.
    If the move scores > threshold with Stockfish, it proceeds; otherwise tries another check move.
    This tests the hypothesis that checking moves alone might be sufficient for puzzle solving.
    """
    
    def __init__(self, score_threshold: float = 1.0, max_attempts: int = 10):
        """
        Initialize the RandomCheckPlayer.
        
        :param score_threshold: Minimum Stockfish score for a move to be accepted
        :param max_attempts: Maximum number of check moves to try before falling back
        """
        self.score_threshold = score_threshold
        self.max_attempts = max_attempts
        self.name = "RandomCheckPlayer"
    
    def get_check_moves(self, board: chess.Board) -> List[str]:
        """
        Get all legal moves that deliver check in the current position.
        
        :param board: Chess board in current position
        :return: List of check moves in UCI format
        """
        check_moves = []
        for move in board.legal_moves:
            temp_board = board.copy()
            temp_board.push(move)
            if temp_board.is_check():
                check_moves.append(move.uci())
        return check_moves
    
    def move(
        self, 
        board: chess.Board, 
        move_limit: int,
        additional_instructions: str | None = None,
        checkmate_retry: bool = False,
        model_settings: dict | None = None,
        max_tries: int = 2,
    ) -> tuple[ChessBM, dict[str, float]]:
        """
        Select a move using the random check strategy.
        
        :param board: Current chess board position
        :param move_limit: Number of moves remaining to solve puzzle
        :param additional_instructions: Unused for this player
        :param checkmate_retry: Unused for this player  
        :param model_settings: Unused for this player
        :param max_tries: Unused for this player
        :return: Tuple of (selected move, moves and scores dict)
        """
        print("==========RANDOM CHECK PLAYER========\n")
        print(f"Current board FEN: {board.fen()}")
        
        check_moves = self.get_check_moves(board)
        moves_and_scores = {}
        
        if not check_moves:
            print("No check moves available, selecting random legal move")
            legal_moves = [move.uci() for move in board.legal_moves]
            if not legal_moves:
                raise ValueError("No legal moves available")
            
            selected_move = random.choice(legal_moves)
            score = score_move_against_stockfish(selected_move, board.fen())
            moves_and_scores[selected_move] = score
            
            return ChessBM(
                reasoning=f"No check moves available, selected random move {selected_move}",
                move=selected_move
            ), moves_and_scores
        
        print(f"Found {len(check_moves)} check moves: {check_moves}")
        
        # Try random check moves until we find one with good score
        attempts = 0
        best_move = None
        best_score = float('-inf')
        
        # Shuffle check moves to ensure randomness
        random.shuffle(check_moves)
        
        for move in check_moves:
            if attempts >= self.max_attempts:
                break
                
            score = score_move_against_stockfish(move, board.fen())
            moves_and_scores[move] = score
            
            print(f"Check move {move} scored: {score}")
            
            # Track best move regardless of threshold
            if score > best_score:
                best_score = score
                best_move = move
            
            # If score exceeds threshold, use this move
            if score > self.score_threshold:
                print(f"Move {move} exceeds threshold ({self.score_threshold}), selecting it")
                return ChessBM(
                    reasoning=f"Selected check move {move} with score {score} > {self.score_threshold}",
                    move=move
                ), moves_and_scores
            
            attempts += 1
        
        # If no move exceeded threshold, use the best scoring check move
        if best_move is not None:
            print(f"No move exceeded threshold, using best check move: {best_move} (score: {best_score})")
            return ChessBM(
                reasoning=f"Used best check move {best_move} with score {best_score} after {attempts} attempts",
                move=best_move
            ), moves_and_scores
        
        # Fallback to random legal move (shouldn't happen if check_moves was not empty)
        legal_moves = [move.uci() for move in board.legal_moves] 
        selected_move = random.choice(legal_moves)
        score = score_move_against_stockfish(selected_move, board.fen())
        moves_and_scores[selected_move] = score
        
        return ChessBM(
            reasoning=f"Fallback to random legal move {selected_move}",
            move=selected_move
        ), moves_and_scores