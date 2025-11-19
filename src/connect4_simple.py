# connect4_simple.py
import math  # For mathematical constants (e.g., infinity)
import random  # For generating random moves (used for opponent)
from game import Game
from search import find_best_move

# ======================
#   MAIN DEMONSTRATION
# ======================
if __name__ == "__main__":
    g = Game()
    human_player = True
    random_opponent = False
    while True:
        print("Board heights:", g.heights)  # Simple board summary

        move, val = find_best_move.find_best_move(g, depth=4)
        if move is None:  # No moves left
            print("No move — draw?")
            break

        print("Engine plays", move, "eval", val)
        g.play(move)

        if g.is_win_for(3 - g.player):  # Check if engine just won
            print("Engine wins!")
            break
        if g.is_draw():  # Draw check
            print("Draw!")
            break

        # --- Random opponent move ---
        if random_opponent:
            opp_moves = g.legal_moves()
            opp = random.choice(opp_moves)  # Pick random legal column
            g.play(opp)
            print("Opponent plays", opp)

        # --- Human player's turn ---
        if human_player:
            print("\nYour turn! (enter a column number 0–6)")
            legal = g.legal_moves()

            while True:
                try:
                    opp = int(input("Choose column: "))
                    if opp not in legal:
                        print("Invalid move. Legal moves are:", legal)
                        continue
                    break
                except ValueError:
                    print("Please enter a valid number.")
            g.play(opp)
            print("You played", opp)

        if g.is_win_for(3 - g.player):  # Opponent win check
            print("Opponent wins!")
            break
        if g.is_draw():
            print("Draw!")
            break
