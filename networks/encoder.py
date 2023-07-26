from enum import Enum
from torch.nn import Module
from torchvision import models


class ENCODER(Enum):
    RESNET_18 = 'RESNET_18'
    RESNET_34 = 'RESNET_34'
    RESNET_50 = 'RESNET_50'
    RESNET_101 = 'RESNET_101'
    RESNET_152 = 'RESNET_152'


class Encoder(Module):
    def __init__(self,
                 encoder_type=ENCODER.RESNET_50,
                 pretrained=False):

        super(Encoder, self).__init__()
        self.encoder = None
        self.features = []
        self.select_encoder_type(encoder_type, pretrained)

    def select_encoder_type(self, encoder_type, pretrained):
        if encoder_type == ENCODER.RESNET_18:
            self.encoder = models.resnet18(pretrained)
        elif encoder_type == ENCODER.RESNET_34:
            self.encoder = models.resnet34(pretrained)
        elif encoder_type == ENCODER.RESNET_50:
            self.encoder = models.resnet50(pretrained)
        elif encoder_type == ENCODER.RESNET_101:
            self.encoder = models.resnet101(pretrained)
        elif encoder_type == ENCODER.RESNET_152:
            self.encoder = models.resnet152(pretrained)
        else:
            raise ValueError("{} is not a valid encoder type".format(encoder_type))

    def forward(self, input_image):
        x = (input_image - 0.45) / 0.225
        self.features.append(self.encoder.relu(self.encoder.bn1(self.encoder.conv1(x))))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features
