from connect4 import Connect4

class AlphaBetaAgent:
    def __init__(self, token: str, max_depth: int = 4):
        self.my_token = token
        self.opponent_token = Connect4.other_token(token)
        self.max_depth = max_depth

    def decide(self, game: 'Connect4') -> int:
        _, best_move = self.alphabeta(game, self.max_depth, float('-inf'), float('inf'), True)
        return best_move

    def _heuristic(self, game: 'Connect4') -> int:
        if game.is_winner(self.my_token):
            return 1
        elif game.is_winner(self.opponent_token):
            return -1
        return 0

    def alphabeta(self, game: 'Connect4', depth: int, alpha: float, beta: float, maximizing_player: bool) -> [float, int]:
        if depth == 0 or game.is_game_over():
            return self._heuristic(game), None

        if maximizing_player:
            max_score = float('-inf')
            best_move = None

            for move in game.possible_drops():
                next_game_state = game.copy()
                next_game_state.drop_token(move)
                score, _ = self.alphabeta(next_game_state, depth - 1, alpha, beta, False)
                if score > max_score:
                    max_score = score
                    best_move = move
                alpha = max(alpha, max_score)
                # Beta cutoff
                if beta <= alpha:
                    break

            return max_score, best_move
        else:
            min_score = float('inf')
            best_move = None

            for move in game.possible_drops():
                next_game_state = game.copy()
                next_game_state.drop_token(move)
                score, _ = self.alphabeta(next_game_state, depth - 1, alpha, beta, True)
                if score < min_score:
                    min_score = score
                    best_move = move
                beta = min(beta, min_score)
                # Alpha cutoff
                if beta <= alpha:
                    break

            return min_score, best_move
