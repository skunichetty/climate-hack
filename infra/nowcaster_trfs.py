import torch
from torchvision.transforms import CenterCrop
from infra.transformer import FourierEncoder2d
from infra.model_base import ModelBase


class SourceEmbedder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(SourceEmbedder, self).__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1, padding="same")
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1, padding="same")
        self.conv3 = torch.nn.Conv2d(128, 128, 3, 1, padding="same")
        self.conv4 = torch.nn.Conv2d(128, 256, 3, 1, padding="same")
        self.conv5 = torch.nn.Conv2d(256, 256, 3, 1, padding="same")
        self.relu = torch.nn.LeakyReLU()
        self.avgpool = torch.nn.AvgPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.linear2 = torch.nn.Linear(1024, 256, bias=True)
        self._init_weights()

    def _init_weights(self):
        weights = (
            self.conv1.weight,
            self.conv2.weight,
            self.conv3.weight,
            self.conv4.weight,
            self.conv5.weight,
            self.linear2.weight,
        )
        biases = (
            self.conv1.bias,
            self.conv2.bias,
            self.conv3.bias,
            self.conv4.bias,
            self.conv5.bias,
            self.linear2.bias,
        )
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="leaky_relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)  # 32x32
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)  # 16x16
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)  # 8x8
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)  # 4x4
        x = self.relu(self.conv5(x))
        x = self.avgpool(self.avgpool(x))  # 2x2
        x = self.flatten(x)  # 1024
        return self.linear2(x)


class TargetEmbedder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TargetEmbedder, self).__init__(*args, **kwargs)
        self.conv1 = torch.nn.Conv2d(1, 64, 3, 1, padding="same")
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1, padding="same")
        self.conv3 = torch.nn.Conv2d(128, 128, 3, 1, padding="same")
        self.conv4 = torch.nn.Conv2d(128, 256, 3, 1, padding="same")
        self.conv5 = torch.nn.Conv2d(256, 256, 3, 1, padding="same")
        self.relu = torch.nn.LeakyReLU()
        self.avgpool = torch.nn.AvgPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.linear2 = torch.nn.Linear(1024, 256, bias=True)
        self._init_weights()

    def _init_weights(self):
        weights = (
            self.conv1.weight,
            self.conv2.weight,
            self.conv3.weight,
            self.conv4.weight,
            self.conv5.weight,
            self.linear2.weight,
        )
        biases = (
            self.conv1.bias,
            self.conv2.bias,
            self.conv3.bias,
            self.conv4.bias,
            self.conv5.bias,
            self.linear2.bias,
        )
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="leaky_relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)  # 32x32
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)  # 16x16
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)  # 8x8
        x = self.relu(self.conv4(x))
        x = self.maxpool(x)  # 4x4
        x = self.relu(self.conv5(x))
        x = self.avgpool(x)  # 2x2
        x = self.flatten(x)  # 1024
        return self.linear2(x)


class Generator(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Generator, self).__init__(*args, **kwargs)
        self.conv1 = torch.nn.ConvTranspose2d(256, 512, 4, 2)
        self.conv2 = torch.nn.ConvTranspose2d(512, 128, 4, 2, padding=1)
        self.conv3 = torch.nn.ConvTranspose2d(128, 64, 4, 2, padding=1)
        self.conv4 = torch.nn.ConvTranspose2d(64, 16, 4, 2, padding=1)
        self.conv5 = torch.nn.ConvTranspose2d(16, 1, 4, 2, padding=1)
        self.relu = torch.nn.LeakyReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        weights = (
            self.conv1.weight,
            self.conv2.weight,
            self.conv3.weight,
            self.conv4.weight,
            self.conv5.weight,
        )
        biases = (
            self.conv1.bias,
            self.conv2.bias,
            self.conv3.bias,
            self.conv4.bias,
            self.conv5.bias,
        )
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="leaky_relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x):
        x = self.relu(self.conv1(x))  # 4x4
        x = self.relu(self.conv2(x))  # 8x8
        x = self.relu(self.conv3(x))  # 16x16
        x = self.relu(self.conv3(x))  # 32x32
        x = self.relu(self.conv3(x))  # 64x64
        return self.sigmoid(x) * 1024.0


class Nowcaster(ModelBase):
    def __init__(self, *args, **kwargs):
        super(Nowcaster, self).__init__(*args, **kwargs)
        # self.sembedder = SourceEmbedder()
        self.embedder = TargetEmbedder()
        self.transformer = torch.nn.Transformer(d_model=256, nhead=4)
        self.generator = Generator()
        self.fourierSource = FourierEncoder2d(0.4, 256, 12)
        self.fourierTarget = FourierEncoder2d(0.4, 256, 24)
        self.center = CenterCrop(64)

    def init_hidden(self, batch_size, in_height, in_width, device):
        h = torch.zeros(
            (batch_size, 64, in_height, in_width),
            device=device,
            dtype=torch.float32,
        )
        c = torch.zeros(
            (batch_size, 64, in_height, in_width),
            device=device,
            dtype=torch.float32,
        )
        return h, c

    def forward(self, src, tgt):
        N, T, H, W = src.shape
        for i in range(N):
            input = self.embedder(self.center(src[i : i + 1].transpose(1, 0)))
            target = self.embedder(tgt[i : i + 1].transpose(1, 0))
            input = self.fourierSource(input)
            target = self.fourierTarget(target)
