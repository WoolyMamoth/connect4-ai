import json
import torch
from torch.utils.data import Dataset, DataLoader
from model import ValueNet
import torch.nn as nn
import torch.optim as optim
import selfplay


class Connect4Dataset(Dataset):
    def __init__(self, file):
        self.samples = []
        with open(file) as f:
            for line in f:
                obj = json.loads(line)
                board = obj["board"]
                value = obj["value"]
                self.samples.append((board, value))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board, value = self.samples[idx]
        board = torch.tensor(board, dtype=torch.float32)
        value = torch.tensor([value], dtype=torch.float32)
        return board, value


# Supervised learining
def train_model(dataset_file="./src/games/dataset.jsonl", epochs=8, batch=256):
    ds = Connect4Dataset(dataset_file)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    model = ValueNet()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for boards, values in dl:
            pred = model(boards)
            loss = loss_fn(pred, values)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dl):.4f}")

    torch.save(model.state_dict(), "valuenet.pt")
    print("Saved model to valuenet.pt")


# Reinforcement learning
def train_model(model, X, y, epochs=5, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()

    return model


if __name__ == "__main__":
    rl = True
    if rl:
        model = ValueNet()
        for i in range(20):
            print(f"=== Iteration {i} ===")
            # 1. generate self-play data
            X, y = selfplay.generate_self_play_data(model, num_games=50)
            # 2. train network
            model = train_model(model, X, y)
    else:
        train_model()
