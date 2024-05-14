import torch.nn as nn


class FcBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=False, **kwargs):
        super().__init__()
        self.ins = in_channels
        self.outs = out_channels
        self.block = self.build_fc(use_norm, **kwargs)

    def build_fc(self, use_norm: bool = False, **kwargs) -> nn.Sequential:
        """
        Builds a fully connected block with the given number of input and output channels.

        Args:
            use_norm (bool, optional): Whether to use batch normalization after the ReLU activation.
            **kwargs: Additional keyword arguments to be passed to the nn.Linear module.

        Returns:
            nn.Sequential: A sequential module containing the fully connected block.
        """
        block = [
            nn.Linear(self.ins, self.outs, bias=False, **kwargs),
            nn.ReLU(),
            nn.BatchNorm1d(self.outs) if use_norm else None,
        ]
        block = [layer for layer in block if layer is not None]

        return nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)
