import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, query, prototypes):
        q = self.q(query)
        k = self.k(prototypes)
        v = self.v(prototypes)
        attn = F.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
        return attn @ v
