import math  # For mathematical constants (e.g., infinity)
import torch
from game import Game
from model import ValueNet

NET = ValueNet()
try:
    NET.load_state_dict(torch.load("valuenet.pt", map_location="cpu"))
except Exception:
    pass
NET.eval()


class search:
    # ======================
    #   EVALUATION FUNCTION
    # ======================
    @staticmethod
    def evaluate(g: Game, me):
        opp = 3 - me  # Opponent’s player number
        score = 0
        center_col = g.getCols() // 2  # Integer division for center column index

        # --- Center control preference ---
        # Encourage occupying the center column
        for r in range(g.getRows()):
            if g.board[r][center_col] == me:
                score += 3
            elif g.board[r][center_col] == opp:
                score -= 3

        # --- Helper: evaluate a 4-cell "window" ---
        def window_score(window):
            my_count = window.count(me)
            opp_count = window.count(opp)

            # Assign scores based on how strong the window is
            if my_count == 4:
                return 1000  # Winning line
            if opp_count == 4:
                return -1000  # Opponent winning line
            if my_count == 3 and opp_count == 0:
                return 50
            if my_count == 2 and opp_count == 0:
                return 10
            if opp_count == 3 and my_count == 0:
                return -80
            if opp_count == 2 and my_count == 0:
                return -5
            return 0  # Neutral window

        B = g.board

        # --- Horizontal windows ---
        for r in range(g.getRows()):
            for c in range(g.getCols() - 3):  # up to COLS-4 index (inclusive)
                w = [B[r][c + i] for i in range(4)]
                score += window_score(w)

        # --- Vertical windows ---
        for c in range(g.getCols()):
            for r in range(g.getRows() - 3):
                w = [B[r + i][c] for i in range(4)]
                score += window_score(w)

        # --- Diagonal down-right (↘) windows ---
        for r in range(g.getRows() - 3):
            for c in range(g.getCols() - 3):
                w = [B[r + i][c + i] for i in range(4)]
                score += window_score(w)

        # --- Diagonal down-left (↙) windows ---
        for r in range(g.getRows() - 3):
            for c in range(3, g.getCols()):
                w = [B[r + i][c - i] for i in range(4)]
                score += window_score(w)

        return score  # Higher = better for 'me'

    @staticmethod
    def evaluate_nn(g, model, me):
        from selfplay import encode_board_state

        board_tensor = torch.tensor(encode_board_state(g), dtype=torch.float32)
        with torch.no_grad():
            val = model(board_tensor).item()
        return val * 1000

    # ======================
    #   NEGAMAX SEARCH
    # ======================
    @staticmethod
    def negamax(g: Game, depth, alpha, beta, color, me):
        # color = +1 if current player == me, -1 otherwise

        # --- Base cases ---
        if g.is_win_for(3 - g.player):  # If previous move won
            return -99999 * color  # Huge negative for losing state
        if g.is_draw() or depth == 0:  # Depth limit or draw
            # return color * find_best_move.evaluate(g, me) # Use heuristic eval
            return color * search.evaluate_nn(g, NET, me)  # Use NN eval

        best = -math.inf  # Initialize best score
        moves = g.get_legal_moves()  # All valid moves

        # Order moves by closeness to center column (good heuristic)
        moves.sort(key=lambda c: -abs(c - g.getCols() // 2))

        # --- Recursive search ---
        for m in moves:
            g.play(m)  # Make move
            # Recursive negamax call — note the sign inversion pattern
            val = -search.negamax(g, depth - 1, -beta, -alpha, -color, me)
            g.undo(m)  # Undo move (restore state)

            if val > best:  # Keep best value
                best = val
            alpha = max(alpha, val)  # Update alpha bound
            if alpha >= beta:  # Beta cutoff (pruning)
                break

        return best  # Best evaluation for this node

    # ======================
    #   MOVE SELECTION
    # ======================
    @staticmethod
    def find_best_move(g: Game, depth):
        me = g.player  # Which player AI is
        best_move = None
        best_val = -math.inf

        # Try every legal move and keep the one with the highest value
        for m in g.get_legal_moves():
            g.play(m)
            val = -search.negamax(g, depth - 1, -math.inf, math.inf, -1, me)
            g.undo(m)
            if val > best_val:
                best_val = val
                best_move = m

        return best_move, best_val  # Return best column and evaluation
