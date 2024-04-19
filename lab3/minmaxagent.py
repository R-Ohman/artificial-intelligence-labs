from connect4 import Connect4

class MinMaxAgent:
    def __init__(self, token: str, max_depth: int = 4):
        self.my_token = token
        self.opponent_token = Connect4.other_token(token)
        self.max_depth = max_depth

    def decide(self, game: 'Connect4') -> int:
        _, best_move = self.minimax(game, self.max_depth, True)
        return best_move

    def _heuristic(self, game: 'Connect4') -> int:
        if game.is_winner(self.my_token):
            return 1
        elif game.is_winner(self.opponent_token):
            return -1
        return 0

    def minimax(self, game: 'Connect4', depth: int, maximizing_player: bool) -> [float, int]:
        if depth == 0 or game.is_game_over():
            return self._heuristic(game), None

        if maximizing_player:
            max_score = float('-inf')
            best_move = None

            for move in game.possible_drops():
                next_game_state = game.copy()
                next_game_state.drop_token(move)
                score, _ = self.minimax(next_game_state, depth - 1, False)
                if score > max_score:
                    max_score = score
                    best_move = move

            return max_score, best_move
        else:
            min_score = float('inf')
            best_move = None

            for move in game.possible_drops():
                next_game_state = game.copy()
                next_game_state.drop_token(move)
                score, _ = self.minimax(next_game_state, depth - 1, True)
                if score < min_score:
                    min_score = score
                    best_move = move

            return min_score, best_move
        