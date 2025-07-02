# NOTE: This script was written with Claude Code 
#!/usr/bin/env python3
"""
Test script to compare RandomCheckPlayer vs LLMPlayer performance on chess puzzles.
This tests the hypothesis that randomly selecting good-scoring check moves 
might be sufficient for puzzle solving.
"""

import os
import random
import chess
import chess.svg
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import concurrent.futures
from dotenv import load_dotenv
import statistics
import numpy as np

from random_check_player import RandomCheckPlayer
from main import PlayGameConfig, play_game, prep_puzzle, count_puzzle_outcomes, load_model
from src.players import CorrectorLLMPlayer, FeedbackModelWithMove
from src.prompts import INSTRUCTIONS_COACH
from src.tools import score_move_against_stockfish
from fire import Fire

load_dotenv()


@dataclass
class GameMetrics:
    """Detailed metrics for a single game"""
    player_name: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves_played: List[str]
    move_scores: List[float]
    total_moves: int
    avg_score: float
    max_score: float
    min_score: float
    score_variance: float
    moves_with_positive_score: int
    moves_with_negative_score: int
    game_duration: float = 0.0  # Could add timing if needed


@dataclass 
class PlayerMetrics:
    """Aggregated metrics for a player across multiple games"""
    player_name: str
    wins: int
    draws: int
    losses: int
    total_games: int
    win_rate: float
    avg_moves_per_game: float
    avg_score_per_move: float
    avg_score_per_game: float
    score_consistency: float  # 1/std_dev of scores
    positive_move_rate: float
    total_moves_played: int
    game_metrics: List[GameMetrics] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """Results comparing two players"""
    random_player_metrics: PlayerMetrics
    llm_player_metrics: PlayerMetrics
    puzzle_fen: str
    mate_in_k: int
    statistical_significance: Dict[str, Any] = field(default_factory=dict)


def create_svg_directory(mate_in_k: int) -> str:
    """Create directory for SVG files"""
    run_id = str(uuid.uuid4())[:8]
    svg_dir = os.path.join("svgs", f"random_check_mate_{mate_in_k}_{run_id}")
    os.makedirs(svg_dir, exist_ok=True)
    return svg_dir


def play_game_with_metrics(
    board: chess.Board,
    player: Any,  # RandomCheckPlayer or LLMPlayer
    config: PlayGameConfig,
    svg_dir: str = None,
    game_number: int = 0
) -> Tuple[str, GameMetrics]:
    """
    Play a single game and collect detailed metrics.
    
    :param board: Chess board with puzzle position
    :param player: Player instance (RandomCheckPlayer or LLMPlayer)
    :param config: Game configuration
    :param svg_dir: Directory to save SVG files (optional)
    :param game_number: Game number for file naming
    :return: Tuple of (game result, game metrics)
    """
    # Create fresh board for thread safety
    game_board = chess.Board(board.fen())
    
    # Initialize metrics tracking
    moves_played = []
    move_scores = []
    player_name = getattr(player, 'name', player.__class__.__name__)
    
    # Create game-specific directory if SVGs requested
    game_dir = None
    if svg_dir:
        game_dir = os.path.join(svg_dir, f"game_{game_number:03d}")
        os.makedirs(game_dir, exist_ok=True)
        
        # Save initial position
        move_idx = 0
        svg_path = os.path.join(game_dir, f"move_{move_idx:03d}_initial.svg")
        with open(svg_path, "w") as f:
            f.write(chess.svg.board(board=game_board))
        move_idx = 1
    
    if config.show_board:
        print(game_board.unicode())

    move_limit = config.move_limit

    while not game_board.is_game_over():
        if move_limit <= 0:
            print("Move limit reached, ending game.")
            break

        # Player move
        if hasattr(player, 'move'):
            move_result = player.move(
                board=game_board,
                move_limit=move_limit,
                checkmate_retry=config.checkmate_retry,
            )
            
            # Handle different return types from different players
            if isinstance(move_result, tuple):
                move, move_scores_dict = move_result
            else:
                move = move_result
                move_scores_dict = {}
        else:
            raise ValueError(f"Player {player_name} does not have a move method")

        # Record the move and calculate its score
        player_move = move.move
        moves_played.append(player_move)
        
        # Get Stockfish score for this move
        move_score = score_move_against_stockfish(player_move, game_board.fen())
        move_scores.append(move_score)

        game_board.push_uci(player_move)
        move_limit -= 1

        if config.show_reasoning:
            print("reasoning:", move.reasoning)
            print(f"{player_name} move:", player_move)
            print(f"Move score: {move_score}")

        if config.show_board:
            print(game_board.unicode())

        # Save SVG after player move
        if game_dir:
            svg_path = os.path.join(game_dir, f"move_{move_idx:03d}_player_{player_move}.svg")
            with open(svg_path, "w") as f:
                f.write(chess.svg.board(board=game_board))
            move_idx += 1

        if game_board.is_game_over():
            break

        # Random opponent move
        from src.players import random_player
        random_move = random_player(game_board)
        game_board.push_uci(random_move)
        
        if config.show_reasoning:
            print(f"Random player move: {random_move}")

        if config.show_board:
            print(game_board.unicode())

        # Save SVG after opponent move
        if game_dir:
            svg_path = os.path.join(game_dir, f"move_{move_idx:03d}_opponent_{random_move}.svg")
            with open(svg_path, "w") as f:
                f.write(chess.svg.board(board=game_board))
            move_idx += 1

    # Save final position if checkmate
    if game_board.is_checkmate() and game_dir:
        svg_path = os.path.join(game_dir, f"move_{move_idx:03d}_final_checkmate.svg")
        with open(svg_path, "w") as f:
            f.write(chess.svg.board(board=game_board))
        
        if game_board.turn == chess.WHITE:
            print("Black wins by checkmate.")
        else:
            print("White wins by checkmate.")

    # Calculate game metrics
    result = game_board.result()
    total_moves = len(moves_played)
    
    if total_moves > 0:
        avg_score = statistics.mean(move_scores)
        max_score = max(move_scores)
        min_score = min(move_scores)
        score_variance = statistics.variance(move_scores) if total_moves > 1 else 0.0
        moves_with_positive_score = sum(1 for score in move_scores if score > 0)
        moves_with_negative_score = sum(1 for score in move_scores if score < 0)
    else:
        avg_score = max_score = min_score = score_variance = 0.0
        moves_with_positive_score = moves_with_negative_score = 0
    
    game_metrics = GameMetrics(
        player_name=player_name,
        result=result,
        moves_played=moves_played,
        move_scores=move_scores,
        total_moves=total_moves,
        avg_score=avg_score,
        max_score=max_score,
        min_score=min_score,
        score_variance=score_variance,
        moves_with_positive_score=moves_with_positive_score,
        moves_with_negative_score=moves_with_negative_score
    )

    return result, game_metrics


def run_player_trials(
    board: chess.Board, 
    player: Any,
    config: PlayGameConfig, 
    number_of_trials: int,
    parallel_games: int = 1,
    save_svgs: bool = True,
    mate_in_k: int = 2
) -> Tuple[List[str], List[GameMetrics]]:
    """
    Run multiple trials with any player and collect detailed metrics.
    
    :param board: Chess board with puzzle position
    :param player: Player instance (RandomCheckPlayer or LLMPlayer)
    :param config: Game configuration
    :param number_of_trials: Number of games to run
    :param parallel_games: Number of parallel games (1 for sequential)
    :param save_svgs: Whether to save SVG files
    :param mate_in_k: Mate in K for directory naming
    :return: Tuple of (game results, game metrics)
    """
    player_name = getattr(player, 'name', player.__class__.__name__)
    
    svg_dir = None
    if save_svgs:
        svg_dir = create_svg_directory(mate_in_k)
        svg_dir = svg_dir.replace("random_check", player_name.lower())
        os.makedirs(svg_dir, exist_ok=True)
        print(f"SVGs will be saved to: {svg_dir}")
    
    def run_single_game(game_num):
        return play_game_with_metrics(board, player, config, svg_dir, game_num)
    
    if parallel_games == 1:
        results = []
        game_metrics = []
        for i in tqdm(range(number_of_trials), desc=f"Running {player_name} games"):
            result, metrics = run_single_game(i)
            results.append(result)
            game_metrics.append(metrics)
    else:
        print(f"Running {number_of_trials} games in parallel with {parallel_games} workers.")
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_games) as executor:
            results_and_metrics = list(
                tqdm(
                    executor.map(run_single_game, range(number_of_trials)),
                    total=number_of_trials,
                    desc=f"Running {player_name} games"
                )
            )
        results = [rm[0] for rm in results_and_metrics]
        game_metrics = [rm[1] for rm in results_and_metrics]
    
    if save_svgs and svg_dir:
        print(f"All SVGs saved to: {svg_dir}")
        print(f"Each game has its own subdirectory with move-by-move SVGs")
    
    return results, game_metrics


def analyze_player_performance(
    results: List[str], 
    game_metrics: List[GameMetrics], 
    fen: str
) -> PlayerMetrics:
    """
    Analyze game results and return comprehensive player statistics.
    
    :param results: List of game results 
    :param game_metrics: List of detailed game metrics
    :param fen: FEN string of the puzzle
    :return: PlayerMetrics with comprehensive statistics
    """
    wins, draws, losses = count_puzzle_outcomes(fen, results)
    total_games = len(results)
    win_rate = wins / total_games if total_games > 0 else 0.0
    
    if not game_metrics:
        return PlayerMetrics(
            player_name="Unknown",
            wins=wins, draws=draws, losses=losses,
            total_games=total_games, win_rate=win_rate,
            avg_moves_per_game=0.0, avg_score_per_move=0.0,
            avg_score_per_game=0.0, score_consistency=0.0,
            positive_move_rate=0.0, total_moves_played=0,
            game_metrics=game_metrics
        )
    
    player_name = game_metrics[0].player_name
    
    # Calculate aggregate statistics
    all_move_scores = []
    total_moves_played = 0
    positive_moves = 0
    
    for metrics in game_metrics:
        all_move_scores.extend(metrics.move_scores)
        total_moves_played += metrics.total_moves
        positive_moves += metrics.moves_with_positive_score
    
    avg_moves_per_game = total_moves_played / total_games if total_games > 0 else 0.0
    avg_score_per_move = statistics.mean(all_move_scores) if all_move_scores else 0.0
    avg_score_per_game = statistics.mean([m.avg_score for m in game_metrics]) if game_metrics else 0.0
    
    # Score consistency (inverse of standard deviation, normalized)
    score_std = statistics.stdev(all_move_scores) if len(all_move_scores) > 1 else 0.0
    score_consistency = 1.0 / (1.0 + score_std) if score_std > 0 else 1.0
    
    positive_move_rate = positive_moves / total_moves_played if total_moves_played > 0 else 0.0
    
    return PlayerMetrics(
        player_name=player_name,
        wins=wins,
        draws=draws, 
        losses=losses,
        total_games=total_games,
        win_rate=win_rate,
        avg_moves_per_game=avg_moves_per_game,
        avg_score_per_move=avg_score_per_move,
        avg_score_per_game=avg_score_per_game,
        score_consistency=score_consistency,
        positive_move_rate=positive_move_rate,
        total_moves_played=total_moves_played,
        game_metrics=game_metrics
    )


def calculate_statistical_significance(
    random_metrics: PlayerMetrics,
    llm_metrics: PlayerMetrics
) -> Dict[str, Any]:
    """
    Calculate statistical significance between two players using t-tests.
    
    :param random_metrics: RandomCheckPlayer metrics
    :param llm_metrics: LLMPlayer metrics
    :return: Dictionary with statistical test results
    """
    try:
        from scipy import stats
        scipy_available = True
    except ImportError:
        print("⚠️  scipy not available, using simplified statistical analysis")
        scipy_available = False
    
    # Extract move scores for statistical testing
    random_scores = []
    llm_scores = []
    
    for game in random_metrics.game_metrics:
        random_scores.extend(game.move_scores)
    
    for game in llm_metrics.game_metrics:
        llm_scores.extend(game.move_scores)
    
    # Basic statistics
    win_rate_diff = llm_metrics.win_rate - random_metrics.win_rate
    
    if scipy_available and len(random_scores) > 1 and len(llm_scores) > 1:
        # Perform t-test on move scores
        t_stat, p_value = stats.ttest_ind(llm_scores, random_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(llm_scores) - 1) * np.var(llm_scores) + 
                             (len(random_scores) - 1) * np.var(random_scores)) / 
                            (len(llm_scores) + len(random_scores) - 2))
        cohens_d = (np.mean(llm_scores) - np.mean(random_scores)) / pooled_std if pooled_std > 0 else 0
        
        # Win rate comparison (binomial test)
        win_rate_p_value = stats.binomtest(
            llm_metrics.wins, 
            random_metrics.total_games, 
            random_metrics.win_rate
        ).pvalue if random_metrics.win_rate > 0 else 1.0
        
    else:
        # Simplified analysis without scipy
        if len(random_scores) > 1 and len(llm_scores) > 1:
            # Simple approximation for t-test
            mean_diff = np.mean(llm_scores) - np.mean(random_scores)
            pooled_var = (np.var(llm_scores) + np.var(random_scores)) / 2
            t_stat = mean_diff / np.sqrt(pooled_var * (1/len(llm_scores) + 1/len(random_scores))) if pooled_var > 0 else 0
            p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + 3)) if abs(t_stat) > 0 else 0.5  # Very rough approximation
            
            pooled_std = np.sqrt(pooled_var)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        else:
            t_stat = p_value = cohens_d = 0
        
        # Simple win rate comparison
        win_rate_p_value = 0.05 if abs(win_rate_diff) > 0.2 else 0.5  # Very simplified
    
    return {
        "move_score_t_stat": t_stat,
        "move_score_p_value": p_value,
        "cohens_d": cohens_d,
        "win_rate_difference": win_rate_diff,
        "win_rate_p_value": win_rate_p_value,
        "significance_level": "significant" if p_value < 0.05 else "not_significant",
        "effect_size": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small",
        "scipy_available": scipy_available
    }


def compare_players(
    random_metrics: PlayerMetrics,
    llm_metrics: PlayerMetrics,
    puzzle_fen: str,
    mate_in_k: int
) -> ComparisonResult:
    """
    Compare two players and return comprehensive comparison results.
    
    :param random_metrics: RandomCheckPlayer metrics
    :param llm_metrics: LLMPlayer metrics  
    :param puzzle_fen: FEN string of the puzzle
    :param mate_in_k: Mate in K moves
    :return: ComparisonResult with detailed comparison
    """
    statistical_significance = calculate_statistical_significance(random_metrics, llm_metrics)
    
    return ComparisonResult(
        random_player_metrics=random_metrics,
        llm_player_metrics=llm_metrics,
        puzzle_fen=puzzle_fen,
        mate_in_k=mate_in_k,
        statistical_significance=statistical_significance
    )


def test_player_comparison(
    mate_in_k: int = 2,
    number_of_trials: int = 20,
    parallel_games: int = 1,
    score_threshold: float = 1.0,
    max_check_attempts: int = 10,
    show_board: bool = False,
    show_reasoning: bool = False,
    save_svgs: bool = True,
    debug_mode: bool = False
) -> ComparisonResult:
    """
    Compare RandomCheckPlayer vs LLMPlayer performance on puzzles.
    
    :param mate_in_k: Mate in K moves puzzle difficulty
    :param number_of_trials: Number of trials to run
    :param parallel_games: Number of parallel games
    :param score_threshold: Stockfish score threshold for move selection
    :param max_check_attempts: Max check moves to try
    :param show_board: Whether to display board states
    :param show_reasoning: Whether to display move reasoning
    :param save_svgs: Whether to save SVG files for animation
    :return: ComparisonResult with performance statistics
    """
    print(f"\n{'='*60}")
    print(f"COMPARING PLAYERS ON MATE-IN-{mate_in_k} PUZZLES")
    print(f"{'='*60}")
    
    # Setup puzzle
    move_limit, fen, board = prep_puzzle(mate_in_k=mate_in_k)
    print(f"Puzzle FEN: {fen}")
    print(f"Move limit: {move_limit}")
    
    if show_board:
        print(f"Initial position:")
        print(board.unicode())
    
    # Game configuration
    config = PlayGameConfig(
        move_limit=move_limit,
        use_plan=False,
        show_board=show_board,
        show_reasoning=show_reasoning,
        checkmate_retry=False,
        svg=False,  # We handle SVGs ourselves
    )
    
    print(f"\nTest settings:")
    print(f"- Score threshold: {score_threshold}")
    print(f"- Max check attempts: {max_check_attempts}")
    print(f"- Number of trials: {number_of_trials}")
    print(f"- Save SVGs: {save_svgs}")
    
    # Create players
    random_player = RandomCheckPlayer(
        score_threshold=score_threshold,
        max_attempts=max_check_attempts
    )
    random_player.name = "RandomCheckPlayer"
    
    # Create LLM player
    model_name, lm = load_model()
    llm_player = CorrectorLLMPlayer(
        model=lm,
        output_type=FeedbackModelWithMove,
        instructions=INSTRUCTIONS_COACH,
        name="CorrectorLLMPlayer",
        retries=3,
    )
    
    print(f"\n🎲 Testing RandomCheckPlayer...")
    random_results, random_metrics = run_player_trials(
        board, random_player, config, number_of_trials, 
        parallel_games, save_svgs, mate_in_k
    )
    random_player_metrics = analyze_player_performance(random_results, random_metrics, fen)
    
    if debug_mode:
        print(f"\n🎲 Debug mode: Testing second RandomCheckPlayer instead of LLM...")
        llm_results, llm_game_metrics = run_player_trials(
            board, random_player, config, number_of_trials, 
            parallel_games, save_svgs, mate_in_k
        )
        llm_player_metrics = analyze_player_performance(llm_results, llm_game_metrics, fen)
    else:
        print(f"\n🤖 Testing LLMPlayer ({model_name})...")
        llm_results, llm_game_metrics = run_player_trials(
            board, llm_player, config, number_of_trials, 
            parallel_games, save_svgs, mate_in_k
        )
        llm_player_metrics = analyze_player_performance(llm_results, llm_game_metrics, fen)
    
    # Compare results
    comparison = compare_players(random_player_metrics, llm_player_metrics, fen, mate_in_k)
    
    return comparison


def print_comparison_results(comparison: ComparisonResult):
    """Print comprehensive comparison results between two players"""
    
    random_metrics = comparison.random_player_metrics
    llm_metrics = comparison.llm_player_metrics
    stats = comparison.statistical_significance
    
    print(f"\n{'='*80}")
    print(f"PLAYER COMPARISON RESULTS - MATE-IN-{comparison.mate_in_k} PUZZLES")
    print(f"{'='*80}")
    
    print(f"Puzzle FEN: {comparison.puzzle_fen}")
    
    # Win Rate Comparison
    print(f"\n📊 WIN RATE COMPARISON:")
    print(f"{'─'*50}")
    print(f"{random_metrics.player_name:20}: {random_metrics.wins:3}/{random_metrics.total_games} ({random_metrics.win_rate:.1%})")
    print(f"{llm_metrics.player_name:20}: {llm_metrics.wins:3}/{llm_metrics.total_games} ({llm_metrics.win_rate:.1%})")
    print(f"{'Difference':20}: {stats['win_rate_difference']:+.1%}")
    
    # Move Quality Comparison
    print(f"\n🎯 MOVE QUALITY COMPARISON (Stockfish Scores):")
    print(f"{'─'*50}")
    print(f"{'Metric':<25} {'Random':<12} {'LLM':<12} {'Difference':<12}")
    print(f"{'─'*50}")
    print(f"{'Avg Score/Move':<25} {random_metrics.avg_score_per_move:8.2f} {llm_metrics.avg_score_per_move:8.2f} {llm_metrics.avg_score_per_move - random_metrics.avg_score_per_move:+8.2f}")
    print(f"{'Avg Score/Game':<25} {random_metrics.avg_score_per_game:8.2f} {llm_metrics.avg_score_per_game:8.2f} {llm_metrics.avg_score_per_game - random_metrics.avg_score_per_game:+8.2f}")
    print(f"{'Positive Move Rate':<25} {random_metrics.positive_move_rate:8.1%} {llm_metrics.positive_move_rate:8.1%} {llm_metrics.positive_move_rate - random_metrics.positive_move_rate:+8.1%}")
    print(f"{'Score Consistency':<25} {random_metrics.score_consistency:8.3f} {llm_metrics.score_consistency:8.3f} {llm_metrics.score_consistency - random_metrics.score_consistency:+8.3f}")
    
    # Efficiency Comparison  
    print(f"\n⚡ EFFICIENCY COMPARISON:")
    print(f"{'─'*50}")
    print(f"{'Avg Moves/Game':<25} {random_metrics.avg_moves_per_game:8.1f} {llm_metrics.avg_moves_per_game:8.1f} {llm_metrics.avg_moves_per_game - random_metrics.avg_moves_per_game:+8.1f}")
    print(f"{'Total Moves Played':<25} {random_metrics.total_moves_played:8d} {llm_metrics.total_moves_played:8d} {llm_metrics.total_moves_played - random_metrics.total_moves_played:+8d}")
    
    # Statistical Significance
    print(f"\n🔬 STATISTICAL ANALYSIS:")
    print(f"{'─'*50}")
    print(f"Move Score T-Statistic:   {stats['move_score_t_stat']:8.3f}")
    print(f"Move Score P-Value:       {stats['move_score_p_value']:8.3f}")
    print(f"Effect Size (Cohen's d):  {stats['cohens_d']:8.3f} ({stats['effect_size']})")
    print(f"Win Rate P-Value:         {stats['win_rate_p_value']:8.3f}")
    print(f"Statistical Significance: {stats['significance_level'].upper()}")
    
    # Summary and Conclusions
    print(f"\n🎯 SUMMARY & CONCLUSIONS:")
    print(f"{'='*50}")
    
    if stats['move_score_p_value'] < 0.05:
        if llm_metrics.avg_score_per_move > random_metrics.avg_score_per_move:
            print(f"✅ LLM significantly outperforms random checking in move quality")
        else:
            print(f"⚠️  Random checking significantly outperforms LLM in move quality")
    else:
        print(f"🤔 No significant difference in move quality between players")
    
    if stats['win_rate_p_value'] < 0.05:
        if llm_metrics.win_rate > random_metrics.win_rate:
            print(f"✅ LLM significantly outperforms random checking in win rate")
        else:
            print(f"⚠️  Random checking significantly outperforms LLM in win rate")
    else:
        print(f"🤔 No significant difference in win rate between players")
    
    # Efficiency analysis
    if llm_metrics.avg_moves_per_game < random_metrics.avg_moves_per_game and llm_metrics.win_rate >= random_metrics.win_rate:
        print(f"⚡ LLM is more efficient (fewer moves to win)")
    elif random_metrics.avg_moves_per_game < llm_metrics.avg_moves_per_game and random_metrics.win_rate >= llm_metrics.win_rate:
        print(f"⚡ Random checking is more efficient (fewer moves to win)")
    
    # Score quality analysis
    if llm_metrics.avg_score_per_move > random_metrics.avg_score_per_move:
        improvement = ((llm_metrics.avg_score_per_move - random_metrics.avg_score_per_move) / abs(random_metrics.avg_score_per_move)) * 100 if random_metrics.avg_score_per_move != 0 else 0
        print(f"🎯 LLM produces {improvement:.1f}% better moves on average")
    
    if llm_metrics.positive_move_rate > random_metrics.positive_move_rate:
        print(f"📈 LLM produces {(llm_metrics.positive_move_rate - random_metrics.positive_move_rate)*100:.1f}% more positive-scoring moves")
    
    # Overall verdict
    print(f"\n🏆 OVERALL VERDICT:")
    print(f"{'─'*30}")
    
    llm_advantages = 0
    random_advantages = 0
    
    if llm_metrics.win_rate > random_metrics.win_rate:
        llm_advantages += 1
    elif random_metrics.win_rate > llm_metrics.win_rate:
        random_advantages += 1
        
    if llm_metrics.avg_score_per_move > random_metrics.avg_score_per_move:
        llm_advantages += 1
    elif random_metrics.avg_score_per_move > llm_metrics.avg_score_per_move:
        random_advantages += 1
        
    if llm_metrics.positive_move_rate > random_metrics.positive_move_rate:
        llm_advantages += 1
    elif random_metrics.positive_move_rate > llm_metrics.positive_move_rate:
        random_advantages += 1
    
    if llm_advantages > random_advantages:
        print(f"🤖 LLM Player demonstrates superior performance overall")
    elif random_advantages > llm_advantages:
        print(f"🎲 Random Check Player demonstrates superior performance overall")
        print(f"💡 This suggests the puzzle may be solvable with simple check-based heuristics")
    else:
        print(f"🤝 Both players perform similarly")
        print(f"💡 This suggests the puzzle difficulty may be at the threshold of random success")


def main(
    mate_in_k: int = 2,
    number_of_trials: int = 10,
    parallel_games: int = 1,
    score_threshold: float = 1.0,
    max_check_attempts: int = 3,
    show_board: bool = False,
    show_reasoning: bool = False,
    save_svgs: bool = False,
    debug_mode: bool = False
):
    """
    Main function to run the player comparison test.
    
    :param mate_in_k: Mate in K moves puzzle difficulty
    :param number_of_trials: Number of trials to run for each player
    :param parallel_games: Number of parallel games
    :param score_threshold: Stockfish score threshold for RandomCheckPlayer
    :param max_check_attempts: Max check moves to try for RandomCheckPlayer
    :param show_board: Whether to display board states
    :param show_reasoning: Whether to display move reasoning
    :param save_svgs: Whether to save SVG files
    """
    print("🔬 CHESS PUZZLE AGENT COMPARISON STUDY")
    print("=" * 65)
    print("Comparing RandomCheckPlayer vs LLMPlayer Performance")
    print("=" * 65)
    
    try:
        comparison = test_player_comparison(
            mate_in_k=mate_in_k,
            number_of_trials=number_of_trials,
            parallel_games=parallel_games,
            score_threshold=score_threshold,
            max_check_attempts=max_check_attempts,
            show_board=show_board,
            show_reasoning=show_reasoning,
            save_svgs=save_svgs,
            debug_mode=debug_mode
        )
        
        print_comparison_results(comparison)
        
        print(f"\n🎯 KEY INSIGHTS:")
        print(f"{'='*50}")
        
        random_metrics = comparison.random_player_metrics
        llm_metrics = comparison.llm_player_metrics
        
        if random_metrics.win_rate > 0.6 and llm_metrics.win_rate > 0.6:
            print(f"📊 Both players solve the puzzle reliably (>60% win rate)")
            if abs(random_metrics.win_rate - llm_metrics.win_rate) < 0.1:
                print(f"💡 Random checking is surprisingly effective for this puzzle!")
                print(f"💭 Consider: The puzzle may be solvable with simple heuristics")
            else:
                better_player = "LLM" if llm_metrics.win_rate > random_metrics.win_rate else "Random"
                print(f"🏆 {better_player} player shows clear superiority")
        
        if llm_metrics.avg_score_per_move > random_metrics.avg_score_per_move:
            print(f"🎯 LLM produces higher quality moves on average")
            print(f"📈 Average score improvement: {llm_metrics.avg_score_per_move - random_metrics.avg_score_per_move:+.2f}")
        
        efficiency_metric = llm_metrics.avg_score_per_move / llm_metrics.avg_moves_per_game if llm_metrics.avg_moves_per_game > 0 else 0
        random_efficiency = random_metrics.avg_score_per_move / random_metrics.avg_moves_per_game if random_metrics.avg_moves_per_game > 0 else 0
        
        if efficiency_metric > random_efficiency:
            print(f"⚡ LLM is more efficient (better score per move ratio)")
        
        print(f"\n📝 RECOMMENDATIONS:")
        print(f"{'─'*40}")
        
        if random_metrics.win_rate > 0.5:
            print(f"• The puzzle may benefit from simpler, check-focused strategies")
            print(f"• Consider hybrid approaches combining LLM reasoning with check heuristics")
        
        if llm_metrics.avg_score_per_move > random_metrics.avg_score_per_move + 1.0:
            print(f"• LLM shows clear move quality advantage - worth the complexity")
        
        if llm_metrics.avg_moves_per_game < random_metrics.avg_moves_per_game:
            print(f"• LLM solves puzzles more efficiently (fewer moves)")
        
        print(f"\n🔬 EXPERIMENT COMPLETE!")
        print(f"Results saved to comparison object for further analysis.")
        
        return comparison
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    Fire(main)  # Use Fire to allow command line arguments
