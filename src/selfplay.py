import random
import json
import torch
from game import Game
import search  # your negamax engine


# ===========================
# SUPERVISED LEARNING HELPERS
# ===========================


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
            move, _ = search.find_best_move(g, depth=engine_depth)

        g.play(move)


def generate_dataset(num_games=2000, output_file="./src/games/dataset.jsonl"):
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

# ==============================
# REINFORCEMENT LEARNING HELPERS
# ==============================


def encode_board(game):
    board = []
    for r in range(6):
        for c in range(7):
            piece = game.board[r][c]
            board.append(piece)
    return torch.tensor(board, dtype=torch.float32)


def self_play_game(model, max_random_moves=5):
    game = Game()
    states = []
    current_player = 1

    while True:
        # store encoded state
        states.append(encode_board(game))

        if game.is_win_for(1):
            return states, 1  # P1 wins → outcome = +1
        if game.is_win_for(2):
            return states, -1  # P2 wins → outcome = -1
        if game.is_draw():
            return states, 0  # draw

        moves = game.get_legal_moves()

        if len(states) <= max_random_moves:
            # exploration phase: choose random move
            move = random.choice(moves)
        else:
            # greedy by neural network value estimate
            best_val = -999
            best_move = random.choice(moves)
            for m in moves:
                g2 = game.clone()
                g2.play(m)
                val = model(encode_board(g2))
                if val > best_val:
                    best_val = val
                    best_move = m
            move = best_move

        game.play(move)


def generate_self_play_data(model, num_games):
    data_states = []
    data_targets = []

    for _ in range(num_games):
        states, outcome = self_play_game(model)
        # outcome is from P1 POV, but states alternate players
        cur_player = 1
        for s in states:
            data_states.append(s)
            data_targets.append(outcome * cur_player)
            cur_player *= -1  # flip perspective each move

    X = torch.stack(data_states)
    y = torch.tensor(data_targets).float().unsqueeze(1)
    return X, y
