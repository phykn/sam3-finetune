import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.d_model = d_model
        self.reset_parameters()

    def reset_parameters(self):
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                if (
                    "box_embed" not in name
                    and "query_embed" not in name
                    and "reference_points" not in name
                ):
                    nn.init.xavier_uniform_(parameter)
