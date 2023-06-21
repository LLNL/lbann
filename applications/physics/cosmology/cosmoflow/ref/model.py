import torch
import torch.nn as nn


class CosmoFlow(nn.Module):
    def __init__(self, input_width=128, batchnorm=False, dropout_rate=0.2):
        super().__init__()

        assert input_width in [128, 256, 512]
        
        Activation = nn.LeakyReLU

        def convbnactpool(in_channels, out_channels, kernel_size=3, padding=1, stride=1, pool=True):
            layers = []
            layers.append(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
            if batchnorm:
                layers.append(nn.BatchNorm3d(out_channels, momentum=1e-3))
            layers.append(Activation())
            if pool:
                layers.append(nn.AvgPool3d(3, 2, 1))
            return layers

        self.conv_layers = []
        self.conv_layers += convbnactpool(4, 16)
        self.conv_layers += convbnactpool(16, 32)
        self.conv_layers += convbnactpool(32, 64)
        self.conv_layers += convbnactpool(64, 128, stride=2)
        self.conv_layers += convbnactpool(128, 256)
        self.conv_layers += convbnactpool(256, 256, pool=(input_width >= 256))
        self.conv_layers += convbnactpool(256, 256, pool=(input_width >= 512))
        self.conv_layers = nn.Sequential(*self.conv_layers)

        def droplinact(in_features, out_features, act=True):
            layers = []
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(in_features, out_features))
            if act:
                layers.append(Activation())
            return layers

        self.lin_layers = []
        self.lin_layers += droplinact(256 * 2**3, 2048)
        self.lin_layers += droplinact(2048, 256)
        self.lin_layers += droplinact(256, 4, act=False)
        self.lin_layers = nn.Sequential(*self.lin_layers)

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.lin_layers(x)
        return x
