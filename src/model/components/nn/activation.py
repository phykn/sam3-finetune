import torch.nn.functional as F


def resolve_activation(name):
    if name == "relu":
        return F.relu
    if name == "gelu":
        return F.gelu
    if name == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {name}.")
