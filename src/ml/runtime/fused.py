import torch

addmm_act_op = torch.ops.aten._addmm_activation


def apply_addmm_activation(activation, linear, tensor):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")

    bias = linear.bias.detach()
    weight = linear.weight.detach()

    bias = bias.to(torch.bfloat16)
    tensor = tensor.to(torch.bfloat16)
    weight = weight.to(torch.bfloat16)

    flat_input = tensor.view(-1, tensor.shape[-1])

    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        output = addmm_act_op(
            bias,
            flat_input,
            weight.t(),
            beta=1,
            alpha=1,
            use_gelu=False,
        )
        return output.view(tensor.shape[:-1] + (output.shape[-1],))

    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        output = addmm_act_op(
            bias,
            flat_input,
            weight.t(),
            beta=1,
            alpha=1,
            use_gelu=True,
        )
        return output.view(tensor.shape[:-1] + (output.shape[-1],))

    raise ValueError(f"Unexpected activation {activation}")
