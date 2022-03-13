from turtle import forward
import torch
from infra.embedders import (
    Generator,
    Embedder,
    ResidualEmbedder,
    ResidualGenerator,
)
from infra.model_base import ModelBase
from random import random
from infra.embedders import Embedder, Generator
from torchvision.transforms import CenterCrop


class Autoencoder(ModelBase):
    def __init__(self, *args, **kwargs):
        super(Autoencoder, self).__init__(*args, **kwargs)
        self.embedder = Embedder()
        self.generator = Generator()

    def forward(self, x):
        return self.generator(self.embedder(x))


class ResidualAutoencoder(ModelBase):
    def __init__(self, *args, **kwargs):
        super(ResidualAutoencoder, self).__init__(*args, **kwargs)
        self.embedder = ResidualEmbedder()
        self.generator = ResidualGenerator()

    def forward(self, x):
        return self.generator(self.embedder(x))


class Encoder(ModelBase):
    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)
        self.lstm = torch.nn.LSTMCell(256, 256)
        self.embedder = Embedder()
        self._init_weights()

    def _init_weights(self):
        weights = (self.lstm.weight_hh, self.lstm.weight_ih)
        biases = (self.lstm.bias_hh, self.lstm.bias_ih)
        for weight in weights:
            torch.nn.init.xavier_normal_(weight)
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x, state):
        h, c = state
        embedding = self.embedder(x)
        h, c = self.lstm(embedding, (h, c))
        return h, c


class Decoder(ModelBase):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)
        self.lstm = torch.nn.LSTMCell(256, 256)
        self.embedder = Embedder()
        self.generator = Generator()
        self._init_weights()

    def _init_weights(self):
        weights = (self.lstm.weight_hh, self.lstm.weight_ih)
        biases = (self.lstm.bias_hh, self.lstm.bias_ih)
        for weight in weights:
            torch.nn.init.xavier_normal_(weight)
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def forward(self, x, state):
        h, c = state
        embedding = self.embedder(x)
        h, c = self.lstm(embedding, (h, c))
        out = self.generator(h)
        return out, (h, c)


class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def init_hidden(self, batch_size):
        hidden_dim = self.encoder.lstm.hidden_size
        device = self.encoder.lstm.weight_hh.device
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

    def forward(self, x, y, cl_thres=0.5):
        # extract shapes
        N, Ti, _, _ = x.shape
        _, To, _, _ = y.shape
        # generate initial encoder state
        he, ce = self.init_hidden(N)
        # forward pass through encoder
        for i in range(Ti):
            he, ce = self.encoder(x[:, i : i + 1], (he, ce))
            he = he.detach()
            ce = ce.detach()
        torch.cuda.empty_cache()
        outputs = []
        start_token = torch.zeros(N, 1, 128, 128, device=he.device)
        outputs.append(start_token)
        for i in range(To):
            rand_val = random()
            if rand_val > cl_thres or i == 0:
                token = outputs[-1]
            else:
                token = y[:, i : i + 1]
            output, (he, ce) = self.decoder(token, (he, ce))
            outputs.append(output)
            he = he.detach()
            ce = ce.detach()
        return torch.cat(outputs[1:], dim=1)


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


class Seq2Seqv2(ModelBase):
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
