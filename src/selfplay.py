import random
import json
from game import Game
from search import find_best_move  # your negamax engine


def encode_board_state(g: Game):
    # two planes: player1, player2 => 84 total features
    planes = []
    for player in [1, 2]:
        for r in range(6):
            for c in range(7):
                planes.append(1 if g.board[r][c] == player else 0)
    return planes  # length = 84


def play_selfplay_game(engine_depth=4):
    g = Game()
    game_history = []  # (board, player_to_move)

    while True:
        # record state
        board = encode_board_state(g)
        game_history.append((board, g.player))

        # check terminal states
        if g.is_win_for(1):
            return game_history, 1
        if g.is_win_for(2):
            return game_history, 2
        if g.is_draw():
            return game_history, 0

        # choose move (mix random for exploration)
        if random.random() < 0.1:  # 10% random noise
            move = random.choice(g.get_legal_moves())
        else:
            move, _ = find_best_move.find_best_move(g, depth=engine_depth)

        g.play(move)


def generate_dataset(num_games=2000, output_file="games/dataset.jsonl"):
    with open(output_file, "w") as f:
        for i in range(num_games):
            history, winner = play_selfplay_game()
            print(f"Game {i+1}/{num_games} Winner: {winner}")

            for board, player in history:
                # outcome relative to the player who was to move
                if winner == 0:
                    outcome = 0
                else:
                    outcome = +1 if winner == player else -1

                f.write(json.dumps({"board": board, "value": outcome}) + "\n")

    print(f"Saved dataset to {output_file}")


if __name__ == "__main__":
    generate_dataset(100, "./src/games/dataset.jsonl")
