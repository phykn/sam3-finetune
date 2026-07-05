import logging

import torch


def compile_components(self):
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 2048

    logging.info("Compiling all components. First time may be very slow.")
    self.maskmem_backbone.forward = torch.compile(
        self.maskmem_backbone.forward,
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )
    self.transformer.encoder.forward = torch.compile(
        self.transformer.encoder.forward,
        mode="max-autotune",
        fullgraph=True,
        dynamic=True,
    )
    self.sam_mask_decoder.forward = torch.compile(
        self.sam_mask_decoder.forward,
        mode="max-autotune",
        fullgraph=True,
        dynamic=False,
    )


def maybe_clone(self, x):
    if self.compile_all_components:
        return x.clone()
    return x
