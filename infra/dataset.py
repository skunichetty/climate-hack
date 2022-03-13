from math import ceil
from typing import Iterator, T_co, Generator, Tuple

from torch.utils.data import IterableDataset
from torchvision.transforms import RandomCrop, CenterCrop
from collections import deque
import pathlib
import torch
import logging
import numpy as np

from memory_profiler import profile


class ClimateHackDataset(IterableDataset):
    def __init__(
        self,
        block_directory,
        device: torch.device,
        count=10000,
        train_mode=True,
        test_split=0.3,
        num_crops=10,
        examples_per_crop=24,
        deterministic=False,
    ) -> None:
        super().__init__()
        self.block_dir = pathlib.Path(block_directory)
        self.deterministic = deterministic
        if deterministic:
            self.crop = CenterCrop(128)
        else:
            self.crop = RandomCrop(128)
        self.device = device
        self.test_split = test_split
        self.train_mode = train_mode
        self.num_crops = num_crops
        self.examples_per_crop = examples_per_crop
        self.final_idx = 500
        if self.train_mode:
            start = 1
            max_idx = (
                self.final_idx - int(ceil(self.test_split * self.final_idx)) - 1
            )
        else:
            start = self.final_idx - int(ceil(self.test_split * self.final_idx))
            max_idx = self.final_idx
        max_examples = (max_idx - start) * self.num_crops
        if count > max_examples:
            logging.warn(
                f"Requested {count} examples but can only produce {max_examples}. "
                "Increase dataset size to produce more samples or increase number of crops per block."
            )
            self.num_examples = max_examples
        else:
            self.num_examples = count
        self.indices = np.random.randint(
            start - 1, max_idx, (self.num_examples,)
        )

    def __len__(self):
        return self.num_examples

    def _load(self, i):
        block = torch.load(self.block_dir / f"block{i}")[0]
        block = block.to(torch.float32)
        block = block.to(self.device)
        block /= 1023
        return block

    def _setup(self, num_examples):
        worker_info = torch.utils.data.get_worker_info()
        if self.train_mode:
            start = 1
            max_idx = (
                self.final_idx - int(ceil(self.test_split * self.final_idx)) - 1
            )
        else:
            start = self.final_idx - int(ceil(self.test_split * self.final_idx))
            max_idx = self.final_idx
        if worker_info:
            n = worker_info.num_workers
            self.id = worker_info.id
            num_examples = int(ceil(num_examples / n))
            num_per_worker = (max_idx - start) // n
            start += num_per_worker * self.id
            max_idx = num_per_worker + start
            indices = self.indices[start - 1 : start - 1 + num_examples]
        else:
            self.id = 0
            indices = self.indices
        logging.debug(
            f"Worker {self.id} initialized - generating {num_examples}"
        )

        return start, num_examples, indices

    def _gen_blocks(self, indices):
        for idx in indices:
            yield self._load(idx)

    def _gen_crops(self, block, examples):
        num_crops = min(
            int(ceil(examples / self.examples_per_crop)), self.num_crops
        )
        count = 0
        for _ in range(num_crops):
            examples_per_crop = min(self.examples_per_crop, examples - count)
            if self.deterministic:
                times = np.arange(12, 36)
            else:
                times = np.random.randint(12, 36, size=(examples_per_crop,))
            crop = self.crop(block)
            for time in times:
                yield crop[time : time + 12], crop[time + 12 : time + 36]
                count += 1

    def _gen_examples(self, start, num_examples, indices):
        count = 0
        block_gen = self._gen_blocks(indices)
        num_crops = self.num_crops
        examples_per_crop = self.examples_per_crop
        while count < num_examples:
            block = next(block_gen)
            start += 1
            examples = min(num_examples - count, num_crops * examples_per_crop)
            crops = self._gen_crops(block, examples)
            for X, y in crops:
                yield X, y
                count += 1
                logging.debug(f"Worker {self.id}: generated example {count}")

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        start, num_examples, indices = self._setup(self.num_examples)
        return iter(self._gen_examples(start, num_examples, indices))


class AutoencoderDataset(IterableDataset):
    def __init__(
        self,
        block_directory,
        device: torch.device,
        count=10000,
        train_mode=True,
        test_split=0.3,
        num_crops=200,
        deterministic=False,
    ) -> None:
        super().__init__()
        self.block_dir = pathlib.Path(block_directory)
        self.deterministic = deterministic
        if deterministic:
            self.crop = CenterCrop(128)
        else:
            self.crop = RandomCrop(128)
        self.device = device
        self.test_split = test_split
        self.train_mode = train_mode
        self.num_crops = num_crops
        self.final_idx = 500
        if self.train_mode:
            start = 1
            max_idx = (
                self.final_idx - int(ceil(self.test_split * self.final_idx)) - 1
            )
        else:
            start = self.final_idx - int(ceil(self.test_split * self.final_idx))
            max_idx = self.final_idx
        max_examples = (max_idx - start) * self.num_crops
        if count > max_examples:
            logging.warn(
                f"Requested {count} examples but can only produce {max_examples}. "
                "Increase dataset size to produce more samples or increase number of crops per block."
            )
            self.num_examples = max_examples
        else:
            self.num_examples = count
        self.indices = np.random.randint(
            start - 1, max_idx, (self.num_examples,)
        )

    def __len__(self):
        return self.num_examples

    def _load(self, i):
        block = torch.load(self.block_dir / f"block{i}")[0]
        block = block.to(torch.float32)
        block = block.to(self.device)
        block /= 1023
        return block

    def _setup(self, num_examples):
        worker_info = torch.utils.data.get_worker_info()
        if self.train_mode:
            start = 1
            max_idx = (
                self.final_idx - int(ceil(self.test_split * self.final_idx)) - 1
            )
        else:
            start = self.final_idx - int(ceil(self.test_split * self.final_idx))
            max_idx = self.final_idx
        if worker_info:
            n = worker_info.num_workers
            self.id = worker_info.id
            num_examples = int(ceil(num_examples / n))
            num_per_worker = (max_idx - start) // n
            start += num_per_worker * self.id
            max_idx = num_per_worker + start
            indices = self.indices[start - 1 : start - 1 + num_examples]
        else:
            self.id = 0
            indices = self.indices
        logging.debug(
            f"Worker {self.id} initialized - generating {num_examples}"
        )

        return start, num_examples, indices

    def _gen_blocks(self, indices):
        for idx in indices:
            yield self._load(idx)

    def _gen_crops(self, block, num_crops):
        for _ in range(num_crops):
            yield self.crop(block)

    def _gen_examples(self, start, num_examples, indices):
        count = 0
        block_gen = self._gen_blocks(indices)
        num_crops = self.num_crops
        while count < num_examples:
            block = next(block_gen)
            start += 1
            num_crops = min(num_examples - count, num_crops)
            crops = self._gen_crops(block, num_crops)
            if self.deterministic:
                times = np.arange(0, 72)
            else:
                times = np.random.randint(12, 60, size=(num_crops,))
            for crop, time in zip(crops, times):
                X = crop[time : time + 1]
                yield X, X
                count += 1
                logging.debug(f"Worker {self.id}: generated example {count}")

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        start, num_examples, indices = self._setup(self.num_examples)
        return iter(self._gen_examples(start, num_examples, indices))
