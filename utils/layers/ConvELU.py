from torch.nn import BatchNorm2d as BatchNormalization
from torch.nn import Conv2d as Convolution
from torch.nn import Module
from torch.nn import ELU
from torch.nn import Sequential


class ConvELU(Module):
    def __init__(self,
                 use_batch_normalization,
                 input_features,
                 output_features,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(ConvELU, self).__init__()

        if use_batch_normalization:
            self.conv_block = Sequential(
                Convolution(input_features,
                            output_features,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=False),
                BatchNormalization(output_features),
                ELU(inplace=True)
            )
        else:
            self.conv_block = Sequential(
                Convolution(input_features,
                            output_features,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=True),
                ELU(inplace=True)
            )

    def forward(self, x):
        return self.conv_block(x)