from math import ceil
from typing import Iterator, T_co

from torch.utils.data import IterableDataset
from torchvision.transforms import CenterCrop, RandomCrop
from collections import deque
import pathlib
import torch
import concurrent.futures
import logging


class ClimateHackDataset(IterableDataset):
    def __init__(self, block_directory, cache_size=16, count=10000) -> None:
        super().__init__()
        self.block_dir = pathlib.Path(block_directory)
        self.config = {
            "cache_size": cache_size,
            "num_examples": count,
            "examples_per_crop": 36,
            "num_crops": 28,
            "max_example_idx": 596,
        }
        assert (
            count
            < self.config["max_example_idx"]
            * self.config["num_crops"]
            * self.config["examples_per_crop"]
        )
        self.window_start = 1
        self.crop = RandomCrop(128)
        self.center = CenterCrop(64)
        self.count = 0
        self.cache = deque([])
        self.id = 0

    def __len__(self):
        return self.config["num_examples"]

    def _load(self, i):
        return torch.load(self.block_dir / f"block{i}")[0].to(torch.float32)

    def _isolate(self, crop, start_idx):
        X = crop[start_idx : start_idx + 12]
        y = self.center(crop[start_idx + 12 : start_idx + 36])
        return X, y

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            n = worker_info.num_workers
            self.id = worker_info.id
            self.config["num_examples"] = int(ceil(self.config["num_examples"] / n))
            self.window_start = (
                1 + (self.config["max_example_idx"] // n) * worker_info.id
            )
            logging.debug(
                f"Worker {self.id} initialized: start - {self.window_start}, count - {self.config['num_examples']}"
            )

        self.cache.appendleft(self._load(self.window_start))
        self.window_start += 1
        while self.count < self.config["num_examples"]:
            block = self.cache[-1]
            crops = [self.crop(block) for _ in range(self.config["num_crops"])]
            for crop in crops:
                for start_time in range(self.config["examples_per_crop"]):
                    X, y = self._isolate(crop, start_time)
                    yield X, y
                    self.count += 1
                    logging.debug(f"Worker {self.id}: wrote example {self.count}")
            self.cache.pop()
            self.cache.appendleft(self._load(self.window_start))
            self.window_start += 1
