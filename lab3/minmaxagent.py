from connect4 import Connect4

class MinMaxAgent:
    def __init__(self, token: str, max_depth: int = 4, use_heuristic: bool = True):
        self.my_token = token
        self.opponent_token = Connect4.other_token(token)
        self.max_depth = max_depth
        self.use_heuristic = use_heuristic

    def decide(self, game: 'Connect4') -> int:
        score, best_move = self.minimax(game, self.max_depth, True)
        print(score)
        return best_move

    def _heuristic(self, game: 'Connect4') -> float:
        heuristic = 0

        # bonus if token is at the center column
        for t in game.center_column():
            if t == self.my_token:
                heuristic += 0.25
            elif t == self.opponent_token:
                heuristic -= 0.1

        three_tokens = ''.join(([self.my_token] * 3))
        two_tokens = ''.join(([self.my_token] * 2))
        has_two_tokens = False
        has_three_tokens = False
        opponent_has_three_tokens = False

        for four in game.iter_fours():
            four_tokens = ''.join(four)
            if three_tokens in four_tokens:
                has_three_tokens = True
                break
            elif two_tokens in four_tokens:
                has_two_tokens = True
            if ''.join([self.opponent_token] * 3) in four_tokens:
                opponent_has_three_tokens = True

        if has_three_tokens:
            heuristic += 0.6
        elif has_two_tokens:
            heuristic += 0.2

        if opponent_has_three_tokens:
            heuristic -= 0.5

        # print(heuristic)
        return heuristic

    def _assessment(self, game: 'Connect4') -> float:
        if game.is_winner(self.my_token):
            return 1
        elif game.is_winner(self.opponent_token):
            return -1
        if not self.use_heuristic:
            return 0
        return self._heuristic(game)

    def minimax(self, game: 'Connect4', depth: int, maximizing_player: bool) -> [float, int]:
        if depth == 0 or game.is_game_over():
            return self._assessment(game), None

        if maximizing_player:
            max_score = float('-inf')
            best_move = None

            for move in game.possible_drops():
                next_game_state = game.copy()
                next_game_state.drop_token(move)
                score, _ = self.minimax(next_game_state, depth - 1, False)
                if score > max_score:
                    # print(score, move)
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
        
