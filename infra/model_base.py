from tabnanny import check
import torch
import os
import uuid
import pathlib
from typing import Union
from functools import reduce


class ModelBase(torch.nn.Module):
    def __init__(self, checkpoint_dir: Union[str, os.PathLike]):
        super(ModelBase, self).__init__()
        if isinstance(checkpoint_dir, str):
            self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        else:
            self.checkpoint_dir = checkpoint_dir
        self._param_count_cache = None

    def save(self, *args, **kwargs):
        """
        Base class file to save model
        """
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)
        id = uuid.uuid4()
        state = {
            "id": id,
            "state_dict": self.state_dict(),
        }
        for key in kwargs:
            state[key] = kwargs[key]
        filename = self.checkpoint_dir / f"{id}.checkpoint.pth.tar"
        torch.save(state, filename)

    def _prettyprint(self, obj) -> str:
        print("TEMP")
        if isinstance(obj, torch.nn.Linear):
            return f"Input Size - {obj.shape[0]}, Output Size - {obj.shape[1]}"
        if isinstance(obj, torch.nn.Linear):
            return f"Input Size - {obj.shape[0]}, Output Size - {obj.shape[1]}"
        return obj.shape

    def summary(self) -> str:
        output = []
        length = 50
        count_condition = self._param_count_cache is None
        if count_condition:
            self._param_count_cache = []

        def extract(input):
            idx, layer = input
            name, module = layer
            if count_condition:
                param_count = sum(map(lambda x: x.numel(), module.parameters()))
                self._param_count_cache.append(param_count)
            else:
                param_count = self._param_count_cache[idx]
            return f"{name}: {module} - {param_count} parameters"

        module_it = self.named_modules()
        next(module_it)
        output.append("Model:")
        output.append("═" * length)
        module_map = map(extract, enumerate(module_it))
        output.append(("\n" + "─" * length + "\n").join(module_map))
        output.append("═" * length)
        if count_condition:
            self.parameter_count = sum(self._param_count_cache)
        output.append(f"Total Parameter Count: {self.parameter_count}")
        output.append("═" * length)
        return "\n".join(output)
