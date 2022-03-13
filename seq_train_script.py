import torch
from infra.model_base import ModelBase
from infra.train import Metrics, Evaluator
from infra.dataset import ClimateHackDataset
from infra.clstm import Seq2Seqv2
from tqdm import tqdm
from loss import MS_SSIMLoss
import matplotlib.pyplot as plt

loss_fn = torch.nn.MSELoss()
seq2seq = Seq2Seqv2(checkpoint_dir="./checkpoint/seq2seq")
optim = torch.optim.Adam(seq2seq.parameters(), lr=1e-2)
checkpoint = torch.load("./checkpoint/autoencoder/basic_autoencoder.pth")
seq2seq.gen.load_state_dict(checkpoint["generator_weight"])
seq2seq.gen.eval()
seq2seq.embedder.load_state_dict(checkpoint["embedder_weight"])
seq2seq.embedder.eval()

for param in seq2seq.embedder.parameters():
    param.requires_grad = False
for param in seq2seq.gen.parameters():
    param.requires_grad = False
