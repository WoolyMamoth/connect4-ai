# connect4_simple.py
import math      # For mathematical constants (e.g., infinity)
import random    # For generating random moves (used for opponent)

# ----- CONSTANTS -----
ROWS = 6         # Number of rows in the Connect 4 board
COLS = 7         # Number of columns
WIN = 4          # Number of consecutive discs needed to win


# ======================
#     GAME CLASS
# ======================
class Game:
    def __init__(self):
        # Create a 6x7 board initialized to 0 (empty)
        # List comprehension: creates 6 lists (rows) each with 7 zeros
        self.board = [[0] * COLS for _ in range(ROWS)]
        
        # Track how many discs are in each column
        # Example: heights[3] = 2 means column 3 has 2 discs placed
        self.heights = [0] * COLS
        
        # Player 1 starts (players are represented as 1 and 2)
        self.player = 1

    # Return a deep copy of the game (for simulation during AI search)
    def clone(self):
        g = Game()
        # Deep copy of each row (row[:] makes a shallow copy of that row)
        g.board = [row[:] for row in self.board]
        g.heights = self.heights[:]  # Copy column heights
        g.player = self.player       # Copy current player
        return g

    # Return a list of all legal column indices that can accept a move
    def legal_moves(self):
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    # Place a disc in the specified column for the current player
    def play(self, col):
        r = self.heights[col]             # Row where the disc will fall
        self.board[r][col] = self.player  # Drop the disc
        self.heights[col] += 1            # Increase column height
        self.player = 3 - self.player     # Toggle player (1 ↔ 2)

    # Undo the last move in a column (used for search / backtracking)
    def undo(self, col):
        self.heights[col] -= 1            # Lower column height
        r = self.heights[col]             # Row that was last filled
        self.board[r][col] = 0            # Remove the disc
        self.player = 3 - self.player     # Toggle back to previous player

    # Check if player p has a winning line on the board
    def is_win_for(self, p):
        B = self.board
        for r in range(ROWS):
            for c in range(COLS):
                if B[r][c] != p: 
                    continue  # Skip if not player p’s disc
                # Horizontal win (→)
                if c + WIN <= COLS and all(B[r][c + i] == p for i in range(WIN)):
                    return True
                # Vertical win (↓)
                if r + WIN <= ROWS and all(B[r + i][c] == p for i in range(WIN)):
                    return True
                # Diagonal down-right win (↘)
                if r + WIN <= ROWS and c + WIN <= COLS and all(B[r + i][c + i] == p for i in range(WIN)):
                    return True
                # Diagonal down-left win (↙)
                if r + WIN <= ROWS and c - WIN + 1 >= 0 and all(B[r + i][c - i] == p for i in range(WIN)):
                    return True
        return False  # No winning line found

    # Check for draw (all columns filled)
    def is_draw(self):
        return all(h == ROWS for h in self.heights)


# ======================
#   EVALUATION FUNCTION
# ======================
def evaluate(g: Game, me):
    opp = 3 - me       # Opponent’s player number
    score = 0
    center_col = COLS // 2  # Integer division for center column index

    # --- Center control preference ---
    # Encourage occupying the center column
    for r in range(ROWS):
        if g.board[r][center_col] == me:
            score += 3
        elif g.board[r][center_col] == opp:
            score -= 3

    # --- Helper: evaluate a 4-cell "window" ---
    def window_score(window):
        my_count = window.count(me)
        opp_count = window.count(opp)

        # Assign scores based on how strong the window is
        if my_count == 4: return 1000       # Winning line
        if opp_count == 4: return -1000     # Opponent winning line
        if my_count == 3 and opp_count == 0: return 50
        if my_count == 2 and opp_count == 0: return 10
        if opp_count == 3 and my_count == 0: return -80
        if opp_count == 2 and my_count == 0: return -5
        return 0                            # Neutral window

    B = g.board

    # --- Horizontal windows ---
    for r in range(ROWS):
        for c in range(COLS - 3):  # up to COLS-4 index (inclusive)
            w = [B[r][c + i] for i in range(4)]
            score += window_score(w)

    # --- Vertical windows ---
    for c in range(COLS):
        for r in range(ROWS - 3):
            w = [B[r + i][c] for i in range(4)]
            score += window_score(w)

    # --- Diagonal down-right (↘) windows ---
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            w = [B[r + i][c + i] for i in range(4)]
            score += window_score(w)

    # --- Diagonal down-left (↙) windows ---
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            w = [B[r + i][c - i] for i in range(4)]
            score += window_score(w)

    return score  # Higher = better for 'me'


# ======================
#   NEGAMAX SEARCH
# ======================
def negamax(g: Game, depth, alpha, beta, color, me):
    # color = +1 if current player == me, -1 otherwise

    # --- Base cases ---
    if g.is_win_for(3 - g.player):  # If previous move won
        return -99999 * color        # Huge negative for losing state
    if g.is_draw() or depth == 0:    # Depth limit or draw
        return color * evaluate(g, me)

    best = -math.inf                # Initialize best score
    moves = g.legal_moves()         # All valid moves

    # Order moves by closeness to center column (good heuristic)
    moves.sort(key=lambda c: -abs(c - COLS // 2))

    # --- Recursive search ---
    for m in moves:
        g.play(m)  # Make move
        # Recursive negamax call — note the sign inversion pattern
        val = -negamax(g, depth - 1, -beta, -alpha, -color, me)
        g.undo(m)  # Undo move (restore state)

        if val > best:              # Keep best value
            best = val
        alpha = max(alpha, val)     # Update alpha bound
        if alpha >= beta:           # Beta cutoff (pruning)
            break

    return best                     # Best evaluation for this node


# ======================
#   MOVE SELECTION
# ======================
def find_best_move(g: Game, depth):
    me = g.player                   # Which player AI is
    best_move = None
    best_val = -math.inf

    # Try every legal move and keep the one with the highest value
    for m in g.legal_moves():
        g.play(m)
        val = -negamax(g, depth - 1, -math.inf, math.inf, -1, me)
        g.undo(m)
        if val > best_val:
            best_val = val
            best_move = m

    return best_move, best_val      # Return best column and evaluation


# ======================
#   MAIN DEMONSTRATION
# ======================
if __name__ == "__main__":
    g = Game()
    while True:
        print("Board heights:", g.heights)   # Simple board summary
        
        move, val = find_best_move(g, depth=4)
        if move is None:                     # No moves left
            print("No move — draw?")
            break

        print("Engine plays", move, "eval", val)
        g.play(move)

        if g.is_win_for(3 - g.player):       # Check if engine just won
            print("Engine wins!")
            break
        if g.is_draw():                      # Draw check
            print("Draw!")
            break

        # --- Random opponent move ---
        opp_moves = g.legal_moves()
        opp = random.choice(opp_moves)       # Pick random legal column
        g.play(opp)
        print("Opponent plays", opp)

        if g.is_win_for(3 - g.player):       # Opponent win check
            print("Opponent wins!")
            break
        if g.is_draw():
            print("Draw!")
            break
