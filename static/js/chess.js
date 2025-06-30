class ChessUI {
    constructor() {
        this.gameState = null;
        this.lastMove = null;
        this.validMoves = [];
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        document.getElementById('start-game').addEventListener('click', () => this.startGame());
        document.getElementById('make-white-move').addEventListener('click', () => this.makeWhiteMove());
        document.getElementById('undo-move').addEventListener('click', () => this.undoMove());
        document.getElementById('new-game-btn').addEventListener('click', () => this.resetGame());
        document.getElementById('make-uci-move').addEventListener('click', () => this.makeUCIMove());
        
        // Enter key on FEN input
        document.getElementById('puzzle-fen').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.startGame();
        });
        
        // Enter key on UCI move input
        document.getElementById('uci-move').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.makeUCIMove();
        });
    }
    
    async startGame() {
        const fenInput = document.getElementById('puzzle-fen');
        const fen = fenInput.value.trim() || "";
        
        try {
            const response = await fetch('/new_game', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fen: fen })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.gameState = data;
                console.log('Game state:', data);
                this.renderBoard(data.board_svg);
                this.updateGameInfo(data);
                this.showGameArea();
                this.loadLegalMoves();
            } else {
                alert('Error starting game: ' + data.error);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    }
    
    async makeWhiteMove() {
        if (!this.gameState || this.gameState.status !== 'waiting_for_white_move') return;
        
        this.showLoading(true);
        document.getElementById('make-white-move').disabled = true;
        
        try {
            const response = await fetch('/make_white_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.gameState = data;
                this.renderBoard(data.board_svg);
                this.updateGameInfo(data);
                this.updateMoveHistory(data.move_history);
                this.lastMove = data.move;
                this.highlightLastMove(data.move);
                
                if (data.status === 'waiting_for_black_move') {
                    this.loadLegalMoves();
                }
            } else {
                alert('Error making white move: ' + data.error);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            this.showLoading(false);
            document.getElementById('make-white-move').disabled = false;
        }
    }
    
    async makeUCIMove() {
        const uciInput = document.getElementById('uci-move');
        const move = uciInput.value.trim();
        
        if (!move) {
            alert('Please enter a move in UCI format (e.g. d5d4)');
            return;
        }
        
        await this.makeBlackMove(move);
        uciInput.value = ''; // Clear the input after making the move
    }
    
    async makeBlackMove(move) {
        if (!this.gameState || this.gameState.status !== 'waiting_for_black_move') return;
        
        try {
            const response = await fetch('/make_black_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ move: move })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.gameState = data;
                this.renderBoard(data.board_svg);
                this.updateGameInfo(data);
                this.updateMoveHistory(data.move_history);
                this.lastMove = data.move;
                this.highlightLastMove(data.move);
                
                if (data.status === 'waiting_for_white_move') {
                    document.getElementById('make-white-move').disabled = false;
                } else if (data.status === 'game_over') {
                    this.showGameResult(data.result);
                }
            } else {
                alert('Error making move: ' + data.error);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    }
    
    async undoMove() {
        if (!this.gameState) return;
        
        try {
            const response = await fetch('/undo_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.gameState = data;
                this.renderBoard(data.board_svg);
                this.updateGameInfo(data);
                this.updateMoveHistory(data.move_history);
                this.clearSelection();
                
                // Update button states
                if (data.status === 'waiting_for_white_move') {
                    document.getElementById('make-white-move').disabled = false;
                    this.loadLegalMoves();
                } else if (data.status === 'waiting_for_black_move') {
                    document.getElementById('make-white-move').disabled = true;
                    this.loadLegalMoves();
                }
            } else {
                alert('Error undoing move: ' + data.error);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    }
    
    async loadLegalMoves() {
        try {
            const response = await fetch('/get_legal_moves');
            const data = await response.json();
            
            if (data.success && data.turn === 'black') {
                this.validMoves = data.legal_moves;
                console.log('Loaded legal moves:', this.validMoves);
            } else {
                this.validMoves = [];
                console.log('No legal moves or not black turn');
            }
        } catch (error) {
            console.error('Error loading legal moves:', error);
            this.validMoves = [];
        }
    }
    
    renderBoard(svgContent) {
        const boardElement = document.getElementById('chess-board');
        boardElement.innerHTML = svgContent;
        
        // Add event listeners to pieces and squares
        this.addBoardEventListeners();
    }
    
    addBoardEventListeners() {
        // No drag and drop needed anymore - using UCI input instead
        const svg = document.querySelector('#chess-board svg');
        if (!svg) return;
        
        // Just make sure the board is visible
        const pieces = svg.querySelectorAll('use[href], use[xlink\\:href]');
        pieces.forEach((piece) => {
            piece.style.cursor = 'default';
            piece.style.pointerEvents = 'none'; // Disable piece interaction
        });
    }
    
    
    isValidMove(move) {
        return this.validMoves.includes(move) || this.validMoves.includes(move + 'q'); // Handle promotion
    }
    
    highlightLastMove(move) {
        // Could implement visual highlighting of the last move if desired
        // For now, just keep it simple
    }
    
    updateGameInfo(data) {
        document.getElementById('turn-indicator').textContent = 
            data.turn === 'white' ? 'White to move' : 'Black to move';
        
        document.getElementById('game-status').textContent = 
            this.getStatusText(data.status);
        
        document.getElementById('current-fen-display').textContent = 
            data.fen || '-';
        
        // Update button states
        const whiteButton = document.getElementById('make-white-move');
        whiteButton.disabled = data.status !== 'waiting_for_white_move';
        
        const undoButton = document.getElementById('undo-move');
        undoButton.disabled = !data.can_undo;
        
        const uciButton = document.getElementById('make-uci-move');
        uciButton.disabled = data.status !== 'waiting_for_black_move';
    }
    
    getStatusText(status) {
        switch (status) {
            case 'waiting_for_white_move': return 'Waiting for White (AI)';
            case 'waiting_for_black_move': return 'Your turn (Black)';
            case 'game_over': return 'Game Over';
            default: return 'Unknown status';
        }
    }
    
    updateMoveHistory(moves) {
        const movesList = document.getElementById('moves-list');
        movesList.innerHTML = '';
        
        for (let i = 0; i < moves.length; i += 2) {
            const moveNumber = Math.floor(i / 2) + 1;
            const whiteMove = moves[i] || '';
            const blackMove = moves[i + 1] || '';
            
            const moveEntry = document.createElement('div');
            moveEntry.className = 'move-entry';
            moveEntry.innerHTML = `
                <span class="move-number">${moveNumber}.</span> 
                ${whiteMove} ${blackMove}
            `;
            movesList.appendChild(moveEntry);
        }
        
        // Scroll to bottom
        movesList.scrollTop = movesList.scrollHeight;
    }
    
    showGameArea() {
        document.getElementById('game-area').style.display = 'flex';
    }
    
    showLoading(show) {
        document.getElementById('loading').style.display = show ? 'block' : 'none';
    }
    
    showGameResult(result) {
        setTimeout(() => {
            alert('Game Over: ' + result);
        }, 100);
    }
    
    resetGame() {
        this.gameState = null;
        this.lastMove = null;
        this.validMoves = [];
        
        document.getElementById('game-area').style.display = 'none';
        document.getElementById('puzzle-fen').value = '';
        document.getElementById('uci-move').value = '';
        document.getElementById('chess-board').innerHTML = '';
        document.getElementById('moves-list').innerHTML = '';
        document.getElementById('current-fen-display').textContent = '-';
    }
}

// Initialize the chess UI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChessUI();
});