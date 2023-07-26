import torch
import torchvision

from torch.nn import BatchNorm2d as BatchNormalization
from torch.nn import Conv2d as Convolution
from torch.nn import ELU
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Sigmoid
from torch.nn import Softmax
from torch.nn.init import kaiming_normal_

from utils.layers import BLOCKS
from utils.layers import ConvELU
from utils.layers import Deconvolution
from utils.layers import ResidualBlock


class BackBone(Module):
    def __init__(self,
                 use_batch_normalization=False,
                 input_features=3,
                 output_features=64):
        super(BackBone, self).__init__()
        self.use_batch_normalization = use_batch_normalization
        self.encoder_channels = [32, 64, 128, 256, 256, 256, 512]
        self.decoder_channels = [256, 128, 128, 128, 64, 64, 32]
        self.encoder = Sequential(
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=input_features,
                    output_features=self.encoder_channels[BLOCKS.FIRST],
                    kernel_size=3),
            ResidualBlock(features=self.encoder_channels[BLOCKS.FIRST]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.FIRST],
                    output_features=self.encoder_channels[BLOCKS.SECOND],
                    padding=1,
                    stride=2),
            ResidualBlock(features=self.encoder_channels[BLOCKS.SECOND]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.SECOND],
                    output_features=self.encoder_channels[BLOCKS.THIRD],
                    padding=1,
                    stride=2),
            ResidualBlock(features=self.encoder_channels[BLOCKS.THIRD]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.THIRD],
                    output_features=self.encoder_channels[BLOCKS.FOURTH],
                    padding=1,
                    stride=2),
            ResidualBlock(features=self.encoder_channels[BLOCKS.FOURTH]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.FOURTH],
                    output_features=self.encoder_channels[BLOCKS.FIFTH],
                    padding=1,
                    stride=2),
            ResidualBlock(features=self.encoder_channels[BLOCKS.FIFTH]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.FIFTH],
                    output_features=self.encoder_channels[BLOCKS.SIXTH],
                    padding=1,
                    stride=2),
            ResidualBlock(features=self.encoder_channels[BLOCKS.SIXTH]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.SIXTH],
                    output_features=self.encoder_channels[BLOCKS.SEVENTH],
                    padding=1,
                    stride=2),
            ResidualBlock(features=self.encoder_channels[BLOCKS.SEVENTH]),
        )

        self.decoder = Sequential(
            Deconvolution(input_features=self.encoder_channels[BLOCKS.SEVENTH],
                          output_features=self.decoder_channels[BLOCKS.FIRST]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.SIXTH] + self.decoder_channels[BLOCKS.FIRST],
                    output_features=self.decoder_channels[BLOCKS.FIRST]),
            Deconvolution(input_features=self.decoder_channels[BLOCKS.FIRST],
                          output_features=self.decoder_channels[BLOCKS.SECOND]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.FIFTH] + self.decoder_channels[BLOCKS.SECOND],
                    output_features=self.decoder_channels[BLOCKS.SECOND]),
            Deconvolution(input_features=self.decoder_channels[BLOCKS.SECOND],
                          output_features=self.decoder_channels[BLOCKS.THIRD]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.FOURTH] + self.decoder_channels[BLOCKS.THIRD],
                    output_features=self.decoder_channels[BLOCKS.THIRD]),
            Deconvolution(input_features=self.decoder_channels[BLOCKS.THIRD],
                          output_features=self.decoder_channels[BLOCKS.FOURTH]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.THIRD] + self.decoder_channels[BLOCKS.FOURTH],
                    output_features=self.decoder_channels[BLOCKS.FOURTH]),
            Deconvolution(input_features=self.decoder_channels[BLOCKS.FOURTH],
                          output_features=self.decoder_channels[BLOCKS.FIFTH]),
            ConvELU(use_batch_normalization=use_batch_normalization,
                    input_features=self.encoder_channels[BLOCKS.SECOND] + self.decoder_channels[BLOCKS.FIFTH],
                    output_features=self.decoder_channels[BLOCKS.FIFTH]),
            Deconvolution(input_features=self.decoder_channels[BLOCKS.FIFTH],
                          output_features=self.decoder_channels[BLOCKS.SIXTH]),
            Convolution(in_channels=self.decoder_channels[BLOCKS.SIXTH] + self.encoder_channels[BLOCKS.FIRST],
                        output_features=output_features,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, Convolution):
                kaiming_normal_(layer.weight.data)
                if layer.bias is not None:
                    layer.bias.data.zero_()
            elif isinstance(layer, BatchNormalization):
                layer.weight.data.fill_(1)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, x):
        encoder_outputs = []
        for index in range(start=0, stop=len(self.encoder), step=2):
            x = self.encoder[index + 1](self.encoder[index](x))
            encoder_outputs.append(x)

        previous_output = encoder_outputs.pop()
        for index in range(start=0, stop=len(self.decoder), step=2):
            corresponding_encoder_output = encoder_outputs.pop()
            out_deconvolutional = self.decoder[index](previous_output, corresponding_encoder_output)
            out_concat = torch.cat((out_deconvolutional, corresponding_encoder_output), 1)
            out_convolutional = self.decoder[index + 1](out_concat)
            previous_output = out_convolutional


class FalNet(Module):
    def __init__(self,
                 use_batch_normalization,
                 height,
                 width,
                 output_channels,
                 disparity_min, disparity_max):
        super(FalNet, self).__init__()
        self.output_channels = output_channels
        self.factor = 1
        self.backbone = BackBone(use_batch_normalization=use_batch_normalization,
                                 input_features=3,
                                 output_features=self.output_channels)
        self.softmax = Softmax(dim=1)
        self.elu = ELU(inplace=True)
        self.sigmoid = Sigmoid()
        self.height = height
        self.width = width
        self.final_convolution = Convolution(self.output_channels,
                                             self.factor * self.output_channels,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0,
                                             bias=True)

        kaiming_normal_(self.final_convolution.weight.data)
        self.final_convolution.bias.data.zero_()
        self.normalize = torchvision.transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
        self.disparity_layered = disparity_max * (disparity_min / disparity_max) ** (
                torch.arange(output_channels) / (output_channels - 1))
        self.disparity_layered = self.disparity_layered[None, :, None, None].expand(-1, -1, self.height,
                                                                                    self.width).cuda()
        self.outputs = {}

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def forward(self, input_left):
        input_left = self.normalize(input_left)
        batch_size, _, _, _ = input_left.shape
        output = self.backbone(input_left)
        self.outputs["logits"] = self.final_convolution(output)
        self.outputs["probability"] = self.softmax(self.outputs["logits"])
        self.outputs["disparity_layered"] = self.disparity_layered.expand(batch_size, -1, -1, -1)
        self.outputs["padding_mask"] = torch.ones_like(self.disparity_layered)
        self.outputs["disparity"] = (self.outputs["probability"] * self.outputs["disparity_layered"]).sum(1, True)
        self.outputs["depth"] = 0.1 * 0.58 * self.width / self.outputs["disparity"]
        return self.outputs
