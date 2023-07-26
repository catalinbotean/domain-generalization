from torch.nn import Conv2d as Convolution
from torch.nn import Module
from torch.nn import ELU
from torch.nn import Sequential


class ResidualBlock(Module):
    def __init__(self, features, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2
        self.residual_block = Sequential(
            Convolution(features,
                        features,
                        kernel_size=kernel_size,
                        padding=self.padding,
                        bias=False),
            ELU(inplace=True),
            Convolution(features,
                        features,
                        kernel_size=kernel_size,
                        padding=self.padding,
                        bias=False),
            ELU(inplace=True)
        )

    def forward(self, x):
        return self.residual_block(x) + x