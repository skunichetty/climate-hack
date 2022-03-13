import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop
from random import random

#########################################
#       Improve this basic model!       #
#########################################


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


class SFF(torch.nn.Module):
    def __init__(self, in_dim, out_dim, *args, **kwargs):
        super(
            SFF,
            self,
        ).__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.dropout(self.linear(x)))


class Seq2Seqv2(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Seq2Seqv2, self).__init__(*args, **kwargs)
        self.embedder = Embedder()
        self.encoder = torch.nn.LSTMCell(512, 256)
        self.decoder = torch.nn.LSTMCell(512, 256)
        self.gen = Generator()
        self.map = SFF(256, 512)
        self.crop = CenterCrop(64)

    def init_hidden(self, batch_size):
        hidden_dim = self.encoder.hidden_size
        device = self.encoder.weight_hh.device
        h = torch.zeros(
            (batch_size, hidden_dim),
            device=device,
            dtype=torch.float32,
        )
        c = torch.zeros(
            (batch_size, hidden_dim),
            device=device,
            dtype=torch.float32,
        )
        return h, c

    def _embed(self, x):
        _, T, _, _ = x.shape
        embeddings = []
        for j in range(T):
            embeddings.append(self.embedder(x[:, j : j + 1]))
        return torch.stack(embeddings, dim=1)

    def _gen(self, x):
        _, T, _ = x.shape
        generated = []
        for j in range(T):
            gen = self.gen(x[:, j])
            if not self.training:
                gen = self.crop(gen)
            generated.append(gen)
        return torch.cat(generated, dim=1)

    def forward_eval(self, x):
        # extract shapes
        N, Ti, _, _ = x.shape
        # embed values
        src_embeddings = self._embed(x)
        # generate initial encoder state
        he, ce = self.init_hidden(N)
        # forward pass through encoder
        for i in range(Ti):
            he, ce = self.encoder(src_embeddings[:, i], (he, ce))
            he = he.detach()
            ce = ce.detach()
        torch.cuda.empty_cache()
        outputs = []
        start_token = torch.zeros(N, 512, device=he.device)
        outputs.append(start_token)
        for i in range(24):
            token = outputs[-1]
            (he, ce) = self.decoder(token, (he, ce))
            output = self.map(he)
            he = he.detach()
            ce = ce.detach()
            outputs.append(output)
        out = self._gen(torch.stack(outputs[1:], dim=1))
        return out * 1023

    def forward_train(self, x, y, cl_thres=0.5):
        # extract shapes
        N, Ti, _, _ = x.shape
        _, To, _, _ = y.shape
        # embed values
        src_embeddings = self._embed(x)
        tgt_embeddings = self._embed(y)
        # generate initial encoder state
        he, ce = self.init_hidden(N)
        # forward pass through encoder
        for i in range(Ti):
            he, ce = self.encoder(src_embeddings[:, i], (he, ce))
            he = he.detach()
            ce = ce.detach()
        torch.cuda.empty_cache()
        outputs = []
        start_token = torch.zeros(N, 512, device=he.device)
        outputs.append(start_token)
        for i in range(To):
            if self.training:
                rand_val = random()
                if rand_val > cl_thres or i == 0:
                    token = outputs[-1]
                else:
                    token = tgt_embeddings[:, i]
            else:
                token = outputs[-1]
            (he, ce) = self.decoder(token, (he, ce))
            output = self.map(he)
            he = he.detach()
            ce = ce.detach()
            outputs.append(output)
        out = self._gen(torch.stack(outputs[1:], dim=1))
        if not self.training:
            out *= 1023
        return out

    def forward(self, x, *args, **kwargs):
        if self.training:
            return self.forward_train(x, kwargs["y"], kwargs["cl_thres"])
        else:
            return self.forward_eval(x)
