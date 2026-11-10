import torch


def evaluate(model, loader):
    scores = []
    for batch in loader:
        scores.append(model(batch).mean().item())
    return sum(scores) / max(1, len(scores))
