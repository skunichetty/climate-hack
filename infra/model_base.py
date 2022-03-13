import torch
import os
import uuid
import pathlib
from typing import Union
from functools import reduce


class ModelBase(torch.nn.Module):
    """
    Extensible layer on top of `torch.nn.Module` to add certain amenities such as
    - Inbuilt model loading and saving
    - A more robust model summary generation (TensorFlow style)
    Note: Only top level modules should extend this class - all other submodules should style defer to `torch.nn.Module`.
    """

    def __init__(
        self, checkpoint_dir: Union[str, os.PathLike], *args, **kwargs
    ):
        """
        Initializes a model.

        Args:
            checkpoint_dir (Union[str, os.PathLike]): Directory to which models should be checkpointed
        """
        super(ModelBase, self).__init__(*args, **kwargs)
        if isinstance(checkpoint_dir, str):
            self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        else:
            self.checkpoint_dir = checkpoint_dir
        self._param_count_cache = None

    def save(self, name: str = None, **metadata):
        """
        Saves model parameters and metadata to file

        Args:
            name (str, optional): Name of model checkpoint. Default is randomly generated UUID.
        """
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True)
        if name is None:
            name = uuid.uuid4()
        state = {
            "id": name,
            "state_dict": self.state_dict(),
        }
        for key in metadata:
            state[key] = metadata[key]
        filename = self.checkpoint_dir / f"{name}.pth"
        torch.save(state, filename)

    def load(self, device: str = "cpu", id: str = None):
        """
        Loads model parameters from file

        Args:
            device (str, optional): Device to load parameters to. Defaults to "cpu".
            id (str, optional): Name of the model checkpoint to load from checkpoint
                                directory. If unspecified, loads the first available model.

        Returns:
            dict: Metadata associated with checkpoint
        """
        if id is None:
            id = next(self.checkpoint_dir.iterdir()).stem
        model_path = self.checkpoint_dir / f"{id}.pth"
        with model_path.open("rb") as input:
            checkpoint = torch.load(input, map_location=device)
            self.load_state_dict(checkpoint["state_dict"])
        return checkpoint

    def summary(self) -> str:
        """
        Generates model summary, specifying all submodules and the parameters associated with each
        submodules.

        Returns:
            str: string model summary
        """
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

        module_it = self.named_children()
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

    def __repr__(self):
        return self.summary()
