import torch
from torch import nn
import segmentation_models_pytorch as smp

class unetvgg(nn.Module):

    def __init__(self,encoder_name='resnet34', encoder_weights=None, clsss=1, activation='softmax', in_channels=3):
        super().__init__()
        # self.args=args
        self.model = smp.Unet(encoder_name=encoder_name,encoder_weights=None,classes=clsss, activation=activation,in_channels=in_channels)

    def forward(self, x):
        return self.model(x)
