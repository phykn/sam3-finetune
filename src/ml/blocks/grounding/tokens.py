from torch import nn


class VisualTokens(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def load_weights(self, ckpt):
        self.register_buffer("language_features", ckpt["language_features"])
        self.register_buffer("language_mask", ckpt["language_mask"])
        self.register_buffer("language_embeds", ckpt.get("language_embeds"))
        return self

    def forward(self):
        out = {
            "language_features": self.language_features,
            "language_mask": self.language_mask,
        }
        if self.language_embeds is not None:
            out["language_embeds"] = self.language_embeds
        return out
