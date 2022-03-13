import torch
from infra.residual import ResidualBasicBlock, ResidualBottleneckBlock


class VAEConvDownBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, bottleneck=True):
        super(VAEConvDownBlock, self).__init__()
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
            block(out_channel, out_channel, activation="leaky_relu", stride=1),
            block(out_channel, out_channel, activation="leaky_relu", stride=1),
        )

    def forward(self, x):
        return self.blocks(self.residual(x))


class VAEConvUpBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, bottleneck=True):
        super(VAEConvUpBlock, self).__init__()
        if bottleneck:
            block = ResidualBottleneckBlock
        else:
            block = ResidualBasicBlock
        self.blocks = torch.nn.Sequential(
            block(in_channel, out_channel, activation="leaky_relu", stride=1),
            block(out_channel, out_channel, activation="leaky_relu", stride=1),
            block(out_channel, out_channel, activation="leaky_relu", stride=1),
            block(out_channel, out_channel, activation="leaky_relu", stride=1),
        )
        self.pool = torch.nn.ConvTranspose2d(in_channel, in_channel, 4, 2, 1)

    def forward(self, x):
        return self.blocks(self.pool(x))


class VAEEncoder(torch.nn.Module):
    pass
