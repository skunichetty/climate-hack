import torch


class SHSAttention(torch.nn.Module):
    def __init__(self, embedding_size=512, output_size=64, mask=False):
        super(SHSAttention, self).__init__()

        self.wquery = torch.nn.Parameter(torch.empty(embedding_size, output_size))
        self.wkey = torch.nn.Parameter(torch.empty(embedding_size, output_size))
        self.wvalue = torch.nn.Parameter(torch.empty(embedding_size, output_size))

        self.softmax = torch.nn.Softmax(dim=1)
        self.register_buffer("output_size", torch.tensor([output_size]))
        self.mask = mask
        self.init_weights()

    def init_weights(self):
        weights = (self.wquery, self.wkey, self.wvalue)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")

    def forward(self, x):
        q, k, v = x @ self.wquery, x @ self.wkey, x @ self.wvalue
        a = q @ k.T
        mask = torch.zeros_like(a) if self.mask else torch.triu(a) * float("-inf")
        return self.softmax((a + mask) / torch.sqrt(self.output_size)) @ v

    def __repr__(self):
        return "SingleHeadAttention"


class SelfAttention(torch.nn.Module):
    def __init__(self, embedding_size=512, hidden_size=64, heads=8, mask=False):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.heads = torch.nn.ModuleList(
            [SHSAttention(embedding_size, hidden_size, mask) for _ in range(heads)]
        )
        self.w0 = torch.nn.Parameter(torch.empty(heads * hidden_size, embedding_size))
        torch.nn.init.kaiming_normal_(self.w0, nonlinearity="relu")

    def forward(self, x):
        embeddings = tuple(map(lambda a: a(x), self.heads))
        concatenated = torch.cat(embeddings, dim=1).to(x.device)
        return concatenated @ self.w0

    def __repr__(self):
        return f"SelfAttention(embedding_dim={self.embedding_size}, hidden_dim={self.hidden_size}, heads={len(self.heads)})"


class SHEDAttention(torch.nn.Module):
    def __init__(self, embedding_size=512, output_size=64):
        super(SHEDAttention, self).__init__()

        self.wquery = torch.nn.Parameter(torch.empty(embedding_size, output_size))
        self.wkey = torch.nn.Parameter(torch.empty(embedding_size, output_size))
        self.wvalue = torch.nn.Parameter(torch.empty(embedding_size, output_size))

        self.softmax = torch.nn.Softmax(dim=1)
        self.register_buffer("output_size", torch.tensor([output_size]))
        self.init_weights()

    def init_weights(self):
        weights = (self.wquery, self.wkey, self.wvalue)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")

    def forward(self, x, hidden):
        q, k, v = x @ self.wquery, hidden @ self.wkey, hidden @ self.wvalue
        return self.softmax((q @ k.T) / torch.sqrt(self.output_size)) @ v

    def __repr__(self):
        return "EncoderDecoderAttention"


class EncDecAttention(torch.nn.Module):
    def __init__(self, embedding_size=512, hidden_size=64, heads=8):
        super(EncDecAttention, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.heads = torch.nn.ModuleList(
            [SHEDAttention(embedding_size, hidden_size) for _ in range(heads)]
        )
        self.w0 = torch.nn.Parameter(torch.empty(heads * hidden_size, embedding_size))
        torch.nn.init.kaiming_normal_(self.w0, nonlinearity="relu")

    def forward(self, x, hidden):
        embeddings = tuple(map(lambda a: a(x, hidden), self.heads))
        concatenated = torch.cat(embeddings, dim=1).to(x.device)
        return concatenated @ self.w0

    def __repr__(self):
        return f"EncoderDecorderAttention(embedding_dim={self.embedding_size}, hidden_dim={self.hidden_size}, heads={len(self.heads)})"


class TransformerNN(torch.nn.Module):
    def __init__(self, size=512):
        super(TransformerNN, self).__init__()
        self.size = size
        self.linear1 = torch.nn.Linear(size, size * 4, bias=True)
        self.linear2 = torch.nn.Linear(size * 4, size, bias=True)
        self.relu = torch.nn.ReLU()
        self.init_weights()

    def init_weights(self):
        weights = (self.linear1.weight, self.linear2.weight)
        biases = (self.linear1.bias, self.linear2.bias)
        for weight in weights:
            torch.nn.init.kaiming_normal_(weight, nonlinearity="relu")
        for bias in biases:
            torch.nn.init.constant_(bias, 0.0)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

    def __repr__(self):
        return f"DenseNN(size={self.size})"


class Encoder(torch.nn.Module):
    def __init__(
        self, embedding_size=512, hidden_size=64, heads=8, enable_pos_encoding=False
    ):
        super(Encoder, self).__init__()
        self.sa = SelfAttention(embedding_size, hidden_size, heads)
        self.ln1 = torch.nn.LayerNorm(embedding_size)
        self.nn = TransformerNN(embedding_size)
        self.ln2 = torch.nn.LayerNorm(embedding_size)
        self.encode_pos = enable_pos_encoding
        self.dropout = torch.nn.Dropout(p=0.1)

    def positional_encoding(self, token_count, embedding_size):
        exp = 2 * (
            torch.div(torch.arange(embedding_size), 2, rounding_mode="trunc")
            / embedding_size
        )
        denom = 10000**exp

        pos_encodings = torch.arange(token_count).view(-1, 1) / denom
        pos_encodings[:, 0::2] = torch.sin(pos_encodings[:, 0::2])
        pos_encodings[:, 1::2] = torch.cos(pos_encodings[:, 1::2])
        return pos_encodings

    def forward(self, x):
        if self.encode_pos:
            x += self.positional_encoding(x.shape[0], x.shape[1])
            x = self.dropout(x)
        x += self.dropout(self.sa(x))
        x = self.ln1(x)
        x += self.dropout(self.nn(x))
        return self.ln2(x)


class Decoder(torch.nn.Module):
    def __init__(
        self,
        embedding_size=512,
        hidden_size=64,
        heads=8,
        enable_pos_encoding=False,
        mask=True,
    ):
        super(Decoder, self).__init__()
        self.sa = SelfAttention(embedding_size, hidden_size, heads, mask=True)
        self.ln1 = torch.nn.LayerNorm(embedding_size)
        self.eda = EncDecAttention(embedding_size, hidden_size, heads)
        self.ln2 = torch.nn.LayerNorm(embedding_size)
        self.nn = TransformerNN(embedding_size)
        self.ln3 = torch.nn.LayerNorm(embedding_size)
        self.encode_pos = enable_pos_encoding
        self.dropout = torch.nn.Dropout(p=0.1)

    def positional_encoding(self, token_count, embedding_size):
        exp = 2 * (
            torch.div(torch.arange(embedding_size), 2, rounding_mode="trunc")
            / embedding_size
        )
        denom = 10000**exp

        pos_encodings = torch.arange(token_count).view(-1, 1) / denom
        pos_encodings[:, 0::2] = torch.sin(pos_encodings[:, 0::2])
        pos_encodings[:, 1::2] = torch.cos(pos_encodings[:, 1::2])
        return pos_encodings

    def forward(self, x, hidden):
        if self.encode_pos:
            x += self.positional_encoding(x.shape[0], x.shape[1])
            x = self.dropout(x)
        x += self.dropout(self.sa(x))
        x = self.ln1(x)
        x += self.dropout(self.eda(x, hidden))
        x = self.ln2(x)
        x += self.dropout(self.nn(x))
        return self.ln3(x)


class Transformer(torch.nn.Module):
    def __init__(
        self,
        enc_count,
        dec_count,
        embedding_size=512,
        hidden_size=64,
        heads=8,
        *args,
        **kwargs,
    ):
        super(Transformer, self).__init__(*args, **kwargs)
        self.enc_stack = torch.nn.ModuleList(
            [Encoder(embedding_size, hidden_size, heads) for _ in range(enc_count)]
        )
        self.enc_stack[0].encode_pos = True
        self.dec_stack = torch.nn.ModuleList(
            [Decoder(embedding_size, hidden_size, heads) for _ in range(dec_count)]
        )
        self.dec_stack[0].encode_pos = True

    def forward(self, x):
        z = x.clone()
        for enc in self.enc_stack:
            z = enc(z)
        o = x.clone()
        for dec in self.dec_stack:
            o = dec(o, z)
        return o
