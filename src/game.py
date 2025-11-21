# ----- CONSTANTS -----
ROWS = 6  # Number of rows in the Connect 4 board
COLS = 7  # Number of columns
WIN = 4  # Number of consecutive discs needed to win


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
        g.player = self.player  # Copy current player
        return g

    # Return a list of all legal column indices that can accept a move
    def get_legal_moves(self):
        return [c for c in range(COLS) if self.heights[c] < ROWS]

    # Place a disc in the specified column for the current player
    def play(self, col):
        r = self.heights[col]  # Row where the disc will fall
        self.board[r][col] = self.player  # Drop the disc
        self.heights[col] += 1  # Increase column height
        self.player = 3 - self.player  # Toggle player (1 ↔ 2)

    # Undo the last move in a column (used for search / backtracking)
    def undo(self, col):
        self.heights[col] -= 1  # Lower column height
        r = self.heights[col]  # Row that was last filled
        self.board[r][col] = 0  # Remove the disc
        self.player = 3 - self.player  # Toggle back to previous player

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
                if (
                    r + WIN <= ROWS
                    and c + WIN <= COLS
                    and all(B[r + i][c + i] == p for i in range(WIN))
                ):
                    return True
                # Diagonal down-left win (↙)
                if (
                    r + WIN <= ROWS
                    and c - WIN + 1 >= 0
                    and all(B[r + i][c - i] == p for i in range(WIN))
                ):
                    return True
        return False  # No winning line found

    # Check for draw (all columns filled)
    def is_draw(self):
        return all(h == ROWS for h in self.heights)

    def getRows(self):
        return ROWS

    def getCols(self):
        return COLS

    def getWin(self):
        return WIN
