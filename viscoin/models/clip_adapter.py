""" 
Model to convert concept space embeddings to clip embeddings
"""

import torch


class ClipAdapter(torch.nn.Module):
    """Basic adapter model with two linear layers"""

    def __init__(self, in_features: int, out_features: int, hidden_size: int = 1024):
        super(ClipAdapter, self).__init__()
        self.linear1 = torch.nn.Linear(in_features, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)
