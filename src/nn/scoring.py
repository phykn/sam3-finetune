import numpy as np
import torch


class DotProductScoring(torch.nn.Module):
    def __init__(
        self,
        d_model,
        d_proj,
        prompt_mlp=None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        super().__init__()
        self.d_proj = d_proj
        assert isinstance(prompt_mlp, torch.nn.Module) or prompt_mlp is None
        self.prompt_mlp = prompt_mlp
        self.prompt_proj = torch.nn.Linear(d_model, d_proj)
        self.hs_proj = torch.nn.Linear(d_model, d_proj)
        self.scale = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val = clamp_max_val

    def mean_pool_text(self, prompt, prompt_mask):
        is_valid = (~prompt_mask).float().permute(1, 0)[..., None]
        num_valid = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)
        return (prompt * is_valid).sum(dim=0) / num_valid

    def forward(self, hs, prompt, prompt_mask):
        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2

        if self.prompt_mlp is not None:
            prompt = self.prompt_mlp(prompt)

        pooled_prompt = self.mean_pool_text(prompt, prompt_mask)
        proj_pooled_prompt = self.prompt_proj(pooled_prompt)
        proj_hs = self.hs_proj(hs)

        scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))
        scores *= self.scale

        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores
