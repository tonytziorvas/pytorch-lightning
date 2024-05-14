import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pool_size=2,
        pool_stride=2,
        use_norm=True,
        **kwargs
    ):
        """
        Initializes the ConvBlock class.

        Args:
            pool_size (int, optional): The size of the max pooling window.
            pool_stride (int, optional): The stride of the max pooling window.
            use_norm (bool, optional): Whether to use batch normalization after the ReLU activation.
            **kwargs: Additional keyword arguments to be passed to the nn.Conv2d module.

        Returns:
            None
        """
        super().__init__()
        self.ins = in_channels
        self.outs = out_channels
        self.block = self.build_conv(use_norm, pool_size, pool_stride, **kwargs)

    def build_conv(
        self, use_norm, pool_size: int = 2, pool_stride: int = 2, **kwargs
    ) -> nn.Sequential:
        """
        Builds a convolutional block with the given number of input and output channels.

        Args:
            use_norm (bool, optional): Whether to use batch normalization after the ReLU activation
            pool_size (int, optional): The size of the max pooling window.
            pool_stride (int, optional): The stride of the max pooling window.
            **kwargs: Additional keyword arguments to be passed to the nn.Conv2d module.

        Returns:
            nn.Sequential: A sequential module containing the convolutional block.
        """

        block = [
            nn.Conv2d(self.ins, self.outs, **kwargs),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride),
            nn.BatchNorm2d(self.outs) if use_norm else None,
        ]

        block = [layer for layer in block if layer is not None]
        return nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)
