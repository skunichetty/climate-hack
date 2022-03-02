from math import ceil
from typing import Iterator, T_co

from torch.utils.data import IterableDataset
import pathlib
import torch
import numpy as np


class ClimateHackDataset(IterableDataset):
    def __init__(self, block_directory, cache_size=16, count=10000) -> None:
        super().__init__()
        self.block_dir = pathlib.Path(block_directory)
        self.cached_blocks = [
            torch.from_numpy(
                np.load(self.block_dir / f"block{i+1}.npy")[0].astype(np.float32)
            )
            for i in range(cache_size)
        ]
        self.cache_size = cache_size
        self.count = count

    def _bounds(self, cx, cy, length=128):
        return cx - length // 2, cx + length // 2, cy - length // 2, cy + length // 2

    def _gen_slice_bounds(self, count=10000):
        centers = np.random.randint(128, 385, (count, 2))
        start = np.random.randint(0, 36, (count, 1))
        return np.hstack((centers, start))

    def fetch_slice(self, block, bound):
        cx, cy, start = bound[0], bound[1], bound[2]
        sx, ex, sy, ey = self._bounds(cx, cy, length=128)
        X = block[start : start + 12, sx:ex, sy:ey]
        sx, ex, sy, ey = self._bounds(cx, cy, length=64)
        y = block[start + 12 : start + 36, sx:ex, sy:ey]
        return X, y

    def __iter__(self) -> Iterator[T_co]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            count = int(ceil(self.count / worker_info.num_workers))
            print(f"Worker {worker_info.id} initialized")
        else:
            count = self.count

        bounds = self._gen_slice_bounds(count)
        block_ids = np.random.randint(0, len(self.cached_blocks), size=(count,))

        return map(
            lambda x: self.fetch_slice(self.cached_blocks[x[0]], x[1]),
            zip(block_ids, bounds),
        )
