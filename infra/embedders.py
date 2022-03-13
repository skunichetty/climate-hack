import torch
from infra.residual import ResidualBasicBlock, ResidualBottleneckBlock, ConvBN


class StridedConvReluBN(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(StridedConvReluBN, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channel, out_channel, kernel_size=4, stride=2, padding=1
        )
        self.bn = torch.nn.BatchNorm2d(out_channel)
        self.relu = torch.nn.LeakyReLU()
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.kaiming_normal_(
            self.conv.weight, nonlinearity="leaky_relu"
        )
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))


class StridedResidual(torch.nn.Module):
    def __init__(self, in_channel, out_channel, bottleneck=True):
        super(StridedResidual, self).__init__()
        if bottleneck:
            block = ResidualBottleneckBlock
        else:
            block = ResidualBasicBlock
        if in_channel != out_channel:
            residual = block(
                in_channel, out_channel, activation="leaky_relu", stride=2
            )
        else:
            residual = block(
                in_channel, out_channel, activation="leaky_relu", stride=1
            )
        self.blocks = torch.nn.Sequential(
            residual,
            block(out_channel, out_channel, activation="leaky_relu", stride=1),
        )

    def forward(self, x):
        return self.blocks(x)


class UStridedConvReluBN(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UStridedConvReluBN, self).__init__()
        self.conv = torch.nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1)
        self.bn = torch.nn.BatchNorm2d(out_channel)
        self.relu = torch.nn.LeakyReLU()
        self.pool = torch.nn.Upsample(scale_factor=2)
        self._init_weights()

    def _init_weights(self):
        torch.nn.init.kaiming_normal_(
            self.conv.weight, nonlinearity="leaky_relu"
        )
        torch.nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))


class UStridedResidual(torch.nn.Module):
    def __init__(self, in_channel, out_channel, bottleneck=True):
        super(UStridedResidual, self).__init__()
        if bottleneck:
            block = ResidualBottleneckBlock
        else:
            block = ResidualBasicBlock
        self.blocks = torch.nn.Sequential(
            block(in_channel, out_channel, activation="leaky_relu", stride=1),
            block(out_channel, out_channel, activation="leaky_relu", stride=1),
        )
        # self.pool = torch.nn.Upsample(scale_factor=2)
        self.pool = torch.nn.ConvTranspose2d(in_channel, in_channel, 4, 2, 1)
        self.bn = torch.nn.BatchNorm2d(in_channel)

    def forward(self, x):
        return self.blocks(self.bn(self.pool(x)))


class Embedder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Embedder, self).__init__(*args, **kwargs)
        self.block1 = StridedConvReluBN(1, 64)
        self.block2 = StridedConvReluBN(64, 64)
        self.block3 = StridedConvReluBN(64, 64)
        self.block4 = StridedConvReluBN(64, 64)
        self.block5 = StridedConvReluBN(64, 64)
        self.block6 = StridedConvReluBN(64, 128)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(512, 512)
        self._init_weights()

    def _init_weights(self):
        weights = (self.linear.weight,)
        biases = (self.linear.bias,)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="leaky_relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return self.linear(self.flatten(x))


class ResidualEmbedder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResidualEmbedder, self).__init__(*args, **kwargs)
        self.conv = ConvBN(
            1, 64, "leaky_relu", padding=1, kernel_size=3, stride=2
        )
        self.activation = torch.nn.LeakyReLU()
        self.mpool = torch.nn.MaxPool2d(2, 2)
        self.apool = torch.nn.AvgPool2d(4, 4)
        self.block1 = StridedResidual(64, 64)
        self.block2 = StridedResidual(64, 128)
        self.block3 = StridedResidual(128, 256)
        self.block4 = StridedResidual(256, 512)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(512, 256)
        self._init_weights()

    def _init_weights(self):
        weights = (self.linear.weight,)
        biases = (self.linear.bias,)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="leaky_relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x):
        x = self.activation(self.conv(x))  # 64 x 64
        x = self.mpool(x)  # 32 x 32
        x = self.block1(x)  # 16 x 16
        x = self.block2(x)  # 8 x 8
        x = self.block3(x)  # 4 x 4
        x = self.block4(x)  # 2 x 2
        x = self.apool(x)  # 1 x 1
        return self.linear(self.flatten(x))


class ResidualGenerator(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResidualGenerator, self).__init__(*args, **kwargs)
        self.block1 = UStridedResidual(256, 256)
        self.block2 = UStridedResidual(256, 128)
        self.block3 = UStridedResidual(128, 128)
        self.block4 = UStridedResidual(128, 64)
        self.block5 = UStridedResidual(64, 64)
        self.block6 = UStridedResidual(64, 16)
        self.block7 = UStridedResidual(16, 4)
        self.channel_conv = torch.nn.Conv2d(4, 1, 1, 1, padding="same")
        self.linear = torch.nn.Linear(256, 256)
        self.relu = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm1d(256)
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self._init_weights()

    def _init_weights(self):
        weights = (self.linear.weight, self.channel_conv.weight)
        biases = (self.linear.bias, self.channel_conv.bias)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="leaky_relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x):
        N, _ = x.shape
        x = self.bn(self.relu(self.linear(x)))
        x = x.reshape(N, 256, 1, 1)
        x = self.block1(x)  # 2 x2
        x = self.block2(x)  # 4 x 4
        x = self.block3(x)  # 4 x 4
        x = self.block4(x)  # 16 x 16
        x = self.block5(x)  # 16 x 16
        x = self.block6(x)  # 32 x 32
        x = self.block7(x)  # 128 x 128
        return self.channel_conv(x)


class Generator(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        self.block1 = UStridedConvReluBN(512, 256)
        self.block2 = UStridedConvReluBN(256, 256)
        self.block3 = UStridedConvReluBN(256, 256)
        self.block4 = UStridedConvReluBN(256, 128)
        self.block5 = UStridedConvReluBN(128, 64)
        self.block6 = UStridedConvReluBN(64, 16)
        self.block7 = UStridedConvReluBN(16, 1)
        self.linear = torch.nn.Linear(512, 512)
        self.relu = torch.nn.LeakyReLU()
        self.bn = torch.nn.BatchNorm1d(512)
        self._init_weights()

    def _init_weights(self):
        weights = (self.linear.weight,)
        biases = (self.linear.bias,)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="leaky_relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x):
        N, _ = x.shape
        x = self.bn(self.relu(self.linear(x)))
        x = x.reshape(N, 512, 1, 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        return x
