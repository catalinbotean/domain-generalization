import torch
from torch.nn import BatchNorm2d as BatchNormalization
from torch.nn import Conv2d as Convolution
from torch.nn import Dropout2d as Dropout
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import Sequential


class ConvNetBlock(Module):
    def __init__(self,
                 initial_features,
                 first_convolution_features,
                 second_convolution_features,
                 dilation_rate,
                 padding,
                 dropout_rate,
                 pre_batch_normalization=True):

        super(ConvNetBlock, self).__init__()

        layers = []
        if pre_batch_normalization:
            layers.append(BatchNormalization(initial_features, momentum=0.0003))

        layers.extend([
            ReLU(inplace=True),
            Convolution(in_channels=initial_features,
                        out_channels=first_convolution_features,
                        kernel_size=1),
            BatchNormalization(first_convolution_features, momentum=0.0003),
            ReLU(inplace=True),
            Convolution(in_channels=first_convolution_features,
                        out_channels=second_convolution_features,
                        kernel_size=3,
                        dilation=dilation_rate,
                        padding=padding)
        ])

        if dropout_rate > 0:
            layers.append(Dropout(dropout_rate))
        self.layers = Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseASPP(Module):
    def __init__(self,
                 features=256,
                 first_convolution_features=512,
                 second_convolution_features=128,
                 dropout_rate=0.1):

        super(DenseASPP, self).__init__()
        self.dilation_rates = [3, 6, 12, 18, 24]
        self.padding = [3, 6, 12, 18, 24]
        self.number_of_blocks = 5

        self.dense_aspp_blocks = ModuleList([
            ConvNetBlock(initial_features=features + second_convolution_features * index,
                         first_convolution_features=first_convolution_features,
                         second_convolution_features=second_convolution_features,
                         dilation_rate=self.dilation_rates[index],
                         padding=self.padding[index],
                         dropout_rate=dropout_rate,
                         pre_batch_normalization=index > 0)
            for index in range(self.number_of_blocks)
        ])

        self.classification = Sequential(
            Dropout(p=dropout_rate),
            Convolution(in_channels=features + second_convolution_features * self.number_of_blocks,
                        out_channels=features,
                        kernel_size=1,
                        padding=0),
        )

    def forward(self, x):
        for dense_aspp_block in self.dense_aspp_blocks:
            x = torch.cat((dense_aspp_block(x), x), dim=1)
        return self.classification(x)
