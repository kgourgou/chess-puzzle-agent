#!/usr/bin/env python3

import chess
import chess.svg
from flask import Flask, render_template, request, jsonify, session
import uuid
from src.players import CorrectorLLMPlayer, FeedbackModelWithMove
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

# Game sessions storage
game_sessions = {}

class GameSession:
    def __init__(self, puzzle_fen):
        self.board = chess.Board(puzzle_fen)
        self.initial_fen = puzzle_fen
        self.move_history = []
        self.board_history = [chess.Board(puzzle_fen)]  # Store board states for undo
        self.status = "waiting_for_white_move"  # white_move, black_move, game_over
        self.result = None
        
        # Initialize CorrectorLLMPlayer for white
        from pydantic_ai.models.openai import OpenAIModel
        model = OpenAIModel(
            model_name=os.getenv("MODEL_NAME", "openai/gpt-4o-mini"),
            provider="openrouter",
        )
        self.white_player = CorrectorLLMPlayer(
            model=model,
            output_type=FeedbackModelWithMove,
            retries=3
        )
    
    def can_undo(self):
        return len(self.move_history) > 0 and self.status != "game_over"
    
    def undo_last_move(self):
        if not self.can_undo():
            return False
        
        # Remove the last move and board state
        self.move_history.pop()
        self.board_history.pop()
        
        # Restore the previous board state
        self.board = self.board_history[-1].copy()
        
        # Update game status
        if self.board.turn == chess.WHITE:
            self.status = "waiting_for_white_move"
        else:
            self.status = "waiting_for_black_move"
        
        self.result = None
        return True

@app.route('/')
def index():
    return render_template('chess.html')

@app.route('/new_game', methods=['POST'])
def new_game():
    data = request.get_json()
    default_fen = "R6R/1r3pp1/4p1kp/3pP3/1r2qPP1/7P/1P1Q3K/8 w - - 1 0"
    fen_input = data.get('fen', '')
    puzzle_fen = fen_input.strip() if fen_input and fen_input.strip() else default_fen
    
    # Create new game session
    session_id = str(uuid.uuid4())
    game_sessions[session_id] = GameSession(puzzle_fen)
    session['game_id'] = session_id
    
    # Get initial board state
    game = game_sessions[session_id]
    board_svg = chess.svg.board(game.board, orientation=chess.BLACK)
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'board_svg': board_svg,
        'fen': game.board.fen(),
        'turn': 'white' if game.board.turn else 'black',
        'status': game.status,
        'can_undo': game.can_undo()
    })

@app.route('/make_white_move', methods=['POST'])
def make_white_move():
    session_id = session.get('game_id')
    if not session_id or session_id not in game_sessions:
        return jsonify({'success': False, 'error': 'No active game session'})
    
    game = game_sessions[session_id]
    
    if game.status != "waiting_for_white_move":
        return jsonify({'success': False, 'error': 'Not white\'s turn'})
    
    def get_white_move():
        # DEBUG MODE: Use a simple predetermined move to save costs
        DEBUG_MODE = False
        
        if DEBUG_MODE:
            try:
                # Just pick the first legal move
                legal_moves = list(game.board.legal_moves)
                if legal_moves:
                    return legal_moves[0]
                return None
            except Exception as e:
                print(f"Error in debug mode: {e}")
                return None
        
        try:
            # Get move from CorrectorLLMPlayer
            move_result, moves_and_scores = game.white_player.move(
                board=game.board,
                move_limit=10  # Set a reasonable move limit
            )
            return chess.Move.from_uci(move_result.move)
        except Exception as e:
            print(f"Error getting white move: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Run async operation in thread pool
    future = executor.submit(get_white_move)
    best_move = future.result(timeout=30)  # 30 second timeout
    
    if best_move is None:
        return jsonify({'success': False, 'error': 'AI failed to generate move'})
    
    # Make the move
    game.board.push(best_move)
    game.move_history.append(str(best_move))
    game.board_history.append(game.board.copy())
    
    # Check game status
    if game.board.is_checkmate():
        game.status = "game_over"
        game.result = "White wins by checkmate"
    elif game.board.is_stalemate() or game.board.is_insufficient_material():
        game.status = "game_over"
        game.result = "Draw"
    else:
        game.status = "waiting_for_black_move"
    
    board_svg = chess.svg.board(game.board, orientation=chess.BLACK)
    
    return jsonify({
        'success': True,
        'move': str(best_move),
        'board_svg': board_svg,
        'fen': game.board.fen(),
        'turn': 'white' if game.board.turn else 'black',
        'status': game.status,
        'result': game.result,
        'move_history': game.move_history,
        'can_undo': game.can_undo()
    })

@app.route('/make_black_move', methods=['POST'])
def make_black_move():
    session_id = session.get('game_id')
    if not session_id or session_id not in game_sessions:
        return jsonify({'success': False, 'error': 'No active game session'})
    
    game = game_sessions[session_id]
    
    if game.status != "waiting_for_black_move":
        return jsonify({'success': False, 'error': 'Not black\'s turn'})
    
    data = request.get_json()
    move_str = data.get('move')
    
    try:
        # Parse and validate move
        move = chess.Move.from_uci(move_str)
        if move not in game.board.legal_moves:
            return jsonify({'success': False, 'error': 'Illegal move'})
        
        # Make the move
        game.board.push(move)
        game.move_history.append(move_str)
        game.board_history.append(game.board.copy())
        
        # Check game status
        if game.board.is_checkmate():
            game.status = "game_over"
            game.result = "Black wins by checkmate"
        elif game.board.is_stalemate() or game.board.is_insufficient_material():
            game.status = "game_over"
            game.result = "Draw"
        else:
            game.status = "waiting_for_white_move"
        
        board_svg = chess.svg.board(game.board, orientation=chess.BLACK)
        
        return jsonify({
            'success': True,
            'move': move_str,
            'board_svg': board_svg,
            'fen': game.board.fen(),
            'turn': 'white' if game.board.turn else 'black',
            'status': game.status,
            'result': game.result,
            'move_history': game.move_history,
            'can_undo': game.can_undo()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Invalid move: {str(e)}'})

@app.route('/undo_move', methods=['POST'])
def undo_move():
    session_id = session.get('game_id')
    if not session_id or session_id not in game_sessions:
        return jsonify({'success': False, 'error': 'No active game session'})
    
    game = game_sessions[session_id]
    
    if not game.can_undo():
        return jsonify({'success': False, 'error': 'Cannot undo - no moves to undo or game is over'})
    
    success = game.undo_last_move()
    
    if success:
        board_svg = chess.svg.board(game.board, orientation=chess.BLACK)
        
        return jsonify({
            'success': True,
            'board_svg': board_svg,
            'fen': game.board.fen(),
            'turn': 'white' if game.board.turn else 'black',
            'status': game.status,
            'result': game.result,
            'move_history': game.move_history,
            'can_undo': game.can_undo()
        })
    else:
        return jsonify({'success': False, 'error': 'Failed to undo move'})

@app.route('/get_legal_moves', methods=['GET'])
def get_legal_moves():
    session_id = session.get('game_id')
    if not session_id or session_id not in game_sessions:
        return jsonify({'success': False, 'error': 'No active game session'})
    
    game = game_sessions[session_id]
    legal_moves = [str(move) for move in game.board.legal_moves]
    
    return jsonify({
        'success': True,
        'legal_moves': legal_moves,
        'turn': 'white' if game.board.turn else 'black'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)