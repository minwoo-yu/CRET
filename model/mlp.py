import torch.nn as nn

def make_model(in_dim, hidden_list, out_dim):
    return MLP(in_dim, hidden_list, out_dim)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_list, out_dim):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU(True))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x)
        return x.view(*shape, -1)