import torch
from torch import nn

PROMPT_IDS = {
    "point": 0,
    "box": 1,
    "mask": 2,
}


class Router(nn.Module):
    def __init__(
        self,
        image_dim: int,
        num_conditions: int,
        num_experts: int,
        hidden_dim: int = 128,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        self.cond = nn.Embedding(num_conditions, embed_dim)
        self.prompt = nn.Embedding(len(PROMPT_IDS), embed_dim)
        self.net = nn.Sequential(
            nn.Linear(image_dim + embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        image: torch.Tensor,
        cond: torch.Tensor,
        prompt: list[str] | tuple[str, ...] | torch.Tensor,
    ) -> torch.Tensor:
        if image.ndim == 4:
            image = image.mean(dim=(2, 3))
        prompt_id = self._prompt_id(prompt, image.device)
        cond = cond.to(device=image.device, dtype=torch.long)
        x = torch.cat([image, self.cond(cond), self.prompt(prompt_id)], dim=1)
        return self.net(x).softmax(dim=1)

    def _prompt_id(
        self,
        prompt: list[str] | tuple[str, ...] | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(prompt, torch.Tensor):
            return prompt.to(device=device, dtype=torch.long)
        return torch.tensor([PROMPT_IDS[item] for item in prompt], device=device)
