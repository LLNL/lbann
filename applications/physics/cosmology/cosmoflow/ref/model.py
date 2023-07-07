import torch
import torch.nn as nn
import math


class CosmoFlow(nn.Module):
    def __init__(
        self, input_width=128, batchnorm=False, dropout_rate=0.5, mlperf=False
    ):
        super().__init__()

        assert input_width in [128, 256, 512]

        if mlperf:
            base_channels = 32
            max_channels = 512
            Pooling = lambda: nn.MaxPool3d(kernel_size=2, stride=2)
        else:
            base_channels = 16
            max_channels = 256
            Pooling = lambda: nn.AvgPool3d(kernel_size=3, stride=2, padding=1)

        Activation = lambda: nn.LeakyReLU(0.3)

        def convbnactpool(in_channels, out_channels):
            layers = []
            layers.append(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=(not batchnorm),
                )
            )
            if batchnorm:
                layers.append(nn.BatchNorm3d(out_channels, momentum=1e-2))
            layers.append(Activation())
            layers.append(Pooling())
            return layers

        in_channels = 4
        self.conv_layers = []
        num_conv_layers = int(math.log2(input_width)) - 2
        for i in range(num_conv_layers):
            out_channels = min(base_channels * 2**i, max_channels)
            self.conv_layers += convbnactpool(in_channels, out_channels)
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*self.conv_layers)

        def linactdrop(in_features, out_features, actdrop=True):
            layers = []
            layers.append(nn.Linear(in_features, out_features))
            if actdrop:
                layers.append(Activation())
                if dropout_rate is not None:
                    layers.append(nn.Dropout(dropout_rate))
            return layers

        self.lin_layers = []
        self.lin_layers += linactdrop(max_channels * 4**3, 128)
        self.lin_layers += linactdrop(128, 64)
        self.lin_layers += linactdrop(64, 4, actdrop=False)
        self.lin_layers = nn.Sequential(*self.lin_layers)

        def initialize_weights(module):
            if isinstance(module, nn.Conv3d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.3)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize_weights)

    def forward(self, x):
        with torch.no_grad():
            x = torch.log1p(x)
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.lin_layers(x)
        x = 1.2 * torch.tanh(x)
        return x
