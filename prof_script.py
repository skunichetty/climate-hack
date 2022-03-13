import torch
from infra.dataset import AutoencoderDataset
from tqdm import tqdm
from memory_profiler import profile
import numpy as np
import cProfile as cprofile


def _load(i):
    block = torch.load(f"temp/processed/block{i}")[0]
    block = block.to(torch.float32)
    block = block.to(torch.device("cpu"))
    block /= 1023
    return block


def _gen_crops(block, num_crops):
    for i in range(num_crops):
        yield block[0].clone()


def main():
    training_data = AutoencoderDataset(
        "./temp/processed",
        device=torch.device("cpu"),
        count=100000,
        train_mode=True,
    )
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=128)
    for (X, y) in tqdm(train_loader):
        pass


main()
