import torch
from src.decoder import PrototypeAttention
from src.episodes import sample_episode


def train(dataset, steps=2000):
    model = PrototypeAttention(dim=256)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(steps):
        support, query = sample_episode(dataset)
        loss = model(support, query).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return model
