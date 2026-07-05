import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + hidden_dims, hidden_dims + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
