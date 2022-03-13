import torch


class ConvBN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, activation, *args, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        torch.nn.init.kaiming_normal_(self.conv.weight, nonlinearity=activation)
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResidualBlockBase(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, activation, stride, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.activation_set = torch.nn.ModuleDict(
            {
                "relu": torch.nn.ReLU(),
                "leaky_relu": torch.nn.LeakyReLU(),
                "gelu": torch.nn.GELU(),
            }
        )
        if in_channels != out_channels:
            self.skip = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
            )
            torch.nn.init.kaiming_normal_(
                self.skip.weight, nonlinearity=activation
            )
            torch.nn.init.constant_(self.skip.bias, 0)
        else:
            self.skip = torch.nn.Identity()
            assert stride == 1
        self.activation = self.activation_set[activation]
        self.residual = torch.nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        skip = self.skip(x)
        out = skip + residual
        return self.activation(out)


class ResidualBasicBlock(ResidualBlockBase):
    def __init__(
        self, in_channels, out_channels, activation, stride, *args, **kwargs
    ):
        super().__init__(
            in_channels, out_channels, activation, stride, *args, **kwargs
        )
        self.residual = torch.nn.Sequential(
            ConvBN(
                in_channels,
                out_channels,
                activation=activation,
                stride=1,
                kernel_size=3,
                padding="same",
            ),
            self.activation,
            ConvBN(
                out_channels,
                out_channels,
                activation=activation,
                stride=stride,
                kernel_size=3,
                padding=1,
            ),
        )


class ResidualBottleneckBlock(ResidualBlockBase):
    def __init__(
        self, in_channels, out_channels, activation, stride, *args, **kwargs
    ):
        super().__init__(
            in_channels, out_channels, activation, stride, *args, **kwargs
        )
        bottlenecked_channels = in_channels // 4
        self.residual = torch.nn.Sequential(
            ConvBN(
                in_channels,
                bottlenecked_channels,
                activation=activation,
                stride=stride,
                kernel_size=1,
            ),
            self.activation,
            ConvBN(
                bottlenecked_channels,
                bottlenecked_channels,
                activation=activation,
                stride=1,
                padding="same",
                kernel_size=3,
            ),
            self.activation,
            ConvBN(
                bottlenecked_channels,
                out_channels,
                activation=activation,
                stride=1,
                padding="same",
                kernel_size=1,
            ),
        )
