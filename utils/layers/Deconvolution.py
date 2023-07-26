from torch.nn import Conv2d as Convolution
from torch.nn import Module
from torch.nn import ELU

class Deconvolution(Module):
    def __init__(self, input_features, output_features):
        super(Deconvolution, self).__init__()
        self.activation_layer = ELU(inplace=True)
        self.convolution_layer = Convolution(input_features,
                                             output_features,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1,
                                             bias=False)

    def forward(self, x, ref):
        x = F.interpolate(x, size=(ref.size(2), ref.size(3)), mode='nearest')
        x = self.activation_layer(self.convolution_layer(x))
        return x