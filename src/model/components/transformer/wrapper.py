import torch.nn as nn


class TransformerWrapper(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        d_model: int,
        two_stage_type="none",
        pos_enc_at_input_dec=True,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.num_queries = decoder.num_queries if decoder is not None else None
        self.pos_enc_at_input_dec = pos_enc_at_input_dec

        assert two_stage_type in ["none"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        self.two_stage_type = two_stage_type

        self._reset_parameters()
        self.d_model = d_model

    def _reset_parameters(self):
        for name, parameter in self.named_parameters():
            if parameter.dim() > 1:
                if (
                    "box_embed" not in name
                    and "query_embed" not in name
                    and "reference_points" not in name
                ):
                    nn.init.xavier_uniform_(parameter)
