from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..components.nn.layers import LayerScale


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = norm_layer(d_model)
        self.ln_2 = norm_layer(d_model)
        self.ls_1 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        self.ls_2 = (
            LayerScale(d_model, ls_init_value)
            if ls_init_value is not None
            else nn.Identity()
        )
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, mlp_width)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(mlp_width, d_model)),
                ]
            )
        )

    def attention(self, q_x, k_x=None, v_x=None, attn_mask=None):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q_x.dtype)
        return self.attn(q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, q_x, k_x=None, v_x=None, attn_mask=None):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        compile_mode: str | None = None,
        use_act_checkpoint: bool = False,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = use_act_checkpoint
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for _ in range(layers)
            ]
        )
        if compile_mode is not None:
            self.forward = torch.compile(
                self.forward,
                mode=compile_mode,
                fullgraph=True,
            )

    def forward(self, x, attn_mask=None):
        for block in self.resblocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, None, None, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask)
        return x


def text_global_pool(x, text=None, pool_type: str = "argmax"):
    if pool_type == "first":
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == "last":
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == "argmax":
        if text is None:
            raise ValueError("text is required for argmax pooling")
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    elif pool_type == "none":
        pooled = tokens = x
    else:
        raise ValueError(f"invalid pool_type: {pool_type}")
    return pooled, tokens


class TextTransformer(nn.Module):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float | None = None,
        output_dim: int = 512,
        no_causal_mask: bool = False,
        pool_type: str = "none",
        proj_bias: bool = False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        output_tokens: bool = False,
        use_ln_post: bool = True,
        compile_mode: str | None = None,
        use_act_checkpoint: bool = False,
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type
        self.token_embedding = nn.Embedding(self.vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            compile_mode=compile_mode,
            use_act_checkpoint=use_act_checkpoint,
        )
        self.ln_final = norm_layer(width) if use_ln_post else nn.Identity()
        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer(
                "attn_mask", self.build_causal_mask(), persistent=False
            )
        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def build_causal_mask(self):
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        return mask.triu_(1)

    def forward(self, text):
        seq_len = text.shape[1]
        x = self.token_embedding(text)
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:seq_len, :seq_len]
        x = x + self.positional_embedding[:seq_len]
        x = self.transformer(x, attn_mask=attn_mask)
        x = self.ln_final(x)
        pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection
        if self.output_tokens:
            return pooled, tokens
        return pooled


class VETextEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        tokenizer,
        width: int = 1024,
        heads: int = 16,
        layers: int = 24,
        context_length: int = 32,
        vocab_size: int = 49408,
        use_ln_post: bool = True,
        compile_mode: str | None = None,
        use_act_checkpoint: bool = True,
    ):
        super().__init__()
        self.context_length = context_length
        self.use_ln_post = use_ln_post
        self.tokenizer = tokenizer
        self.encoder = TextTransformer(
            context_length=self.context_length,
            vocab_size=vocab_size,
            width=width,
            heads=heads,
            layers=layers,
            output_tokens=True,
            use_ln_post=use_ln_post,
            compile_mode=compile_mode,
            use_act_checkpoint=use_act_checkpoint,
        )
        self.resizer = nn.Linear(self.encoder.width, d_model)

    def forward(self, text, input_boxes=None, device=None):
        if isinstance(text[0], str):
            if input_boxes is not None and len(input_boxes) != 0:
                raise AssertionError("input_boxes are not supported by VETextEncoder")
            tokenized = self.tokenizer(text, context_length=self.context_length).to(
                device
            )
            text_attention_mask = (tokenized != 0).bool()
            inputs_embeds = self.encoder.token_embedding(tokenized)
            _, text_memory = self.encoder(tokenized)
            assert text_memory.shape[1] == inputs_embeds.shape[1]
            text_attention_mask = text_attention_mask.ne(1)
            text_memory = text_memory.transpose(0, 1)
            text_memory_resized = self.resizer(text_memory)
        else:
            text_attention_mask, text_memory_resized, tokenized = text
            inputs_embeds = tokenized["inputs_embeds"]
            if input_boxes is not None and len(input_boxes) != 0:
                raise AssertionError(
                    "can't replace boxes in text if it's already encoded"
                )
        return (
            text_attention_mask,
            text_memory_resized,
            inputs_embeds.transpose(0, 1),
        )
