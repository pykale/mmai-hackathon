"""
Base dataset and dataloader utilities for custom and graph data.

The goal is to have easy to extend dataset class for various modalities that
can also be combined to obtain multimodal datasets.

We provided two base classes, but feel free to modify them as needed.

Classes:
    BaseDataset: Template for custom datasets, supports multimodal aggregation.
    BaseDataLoader: Template for custom dataloaders based on torch_geometric.data.DataLoader for graph/non-graph batching.
    BaseSampler: Template for custom samplers, e.g., for multimodal sampling.
"""

from torch.utils.data import Dataset, Sampler
from torch_geometric.data import DataLoader

__all__ = ["BaseDataset", "BaseDataLoader", "BaseSampler"]


class BaseDataset(Dataset):
    """
    Template base class for building datasets.

    Subclasses must implement `__len__` and `__getitem__`. Optionally override `extra_repr()`
    and `__add__()` (for multimodal aggregation) if needed. `prepare_data()` can be used
    as a class method to handle data downloading, preprocessing, and splitting if necessary.

    Args:
        *args: Positional arguments for dataset initialization.
        **kwargs: Keyword arguments for dataset initialization.

    Initial Idea:
        Support composing modality-specific datasets via the `+` operator, e.g.,
        `mm_ds = ecg_ds + image_ds [+ text_ds]`. Subclasses implementing `__add__`
        should align samples (by index/ID) and return a combined dataset.
        Note: This is not a strict requirement, just a starting idea you can adapt or improve.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("BaseDataset is an abstract class and cannot be instantiated directly.")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__ method.")

    def __getitem__(self, idx: int):
        """Return a single sample from the dataset."""
        raise NotImplementedError("Subclasses must implement __getitem__ method.")

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self) -> str:
        """Return any extra information about the dataset."""
        return f"sample_size={len(self)}"

    def __add__(self, other):
        """
        Combine with another dataset.

        Override in subclasses to implement multimodal aggregation.

        Args:
            other: Another dataset to combine with this one.

        Initial Idea:
            Use `__add__` to align and merge heterogeneous modalities into a single
            dataset, keeping shared IDs synchronized.
            Note: This is not mandatory; treat it as a sketch you can refine or replace.
        """
        raise NotImplementedError("Subclasses may implement __add__ method if needed.")

    @classmethod
    def prepare_data(cls, *args, **kwargs):
        """
        Prepare data for the dataset. Possible use case:
        1. Downloading data from a remote source.
        2. Preprocessing raw data into a format suitable for the dataset.
        3. Any other setup tasks required before the dataset can be used. An example
            could be dataset subsetting to train/val/test splits.
        4. Returns the dataset object given the prepared data and available splits.

        You may skip this method if you feel that it is not necessary for your ideal use case.

        Args:
            *args: Positional arguments for data preparation.
            **kwargs: Keyword arguments for data preparation.

        Returns:
            Union[BaseDataset, Dict[str, BaseDataset]]: The prepared dataset or a dictionary
            of datasets for different splits (e.g., train, val, test).
        """
        raise NotImplementedError("Subclasses may implement prepare_data class method if needed.")


class CXRDataset(BaseDataset):
    """Example subclass for a chest X-ray dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ECGDataset(BaseDataset):
    """Example subclass for an ECG dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BaseDataLoader(DataLoader):
    """
    DataLoader for graph and non-graph data.

    Directly inherits from `torch_geometric.data.DataLoader`. Use it like
    `torch.utils.data.DataLoader`.

    Args:
        dataset (BaseDataset): The dataset from which to load data.
        batch_size (int): How many samples per batch to load. Default: 1.
        shuffle (bool): Whether to reshuffle the data at every epoch. Default: False.
        follow_batch (list): Creates assignment batch vectors for each key in the list. Default: None.
        exclude_keys (list): Keys to exclude. Default: None.
        **kwargs: Additional arguments forwarded to `torch.utils.data.DataLoader`.

    Initial Idea:
        A future `MultimodalDataLoader` can accept a tuple of modality datasets and yield
        batches like `{"ecg": ..., "image": ...}`. Missing modalities are simply absent
        in that batch, keeping iteration simple and robust.
        Note: This is not a hard requirement. Consider it a future-facing idea you can evolve.
    """


class MultimodalDataLoader(BaseDataLoader):
    """Example dataloader for handling multiple data modalities."""

    def __init__(self, data_list, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_list = data_list


class BaseSampler(Sampler):
    """
    Base sampler to extend for custom sampling strategies.

    Args:
        data_source (Sized): The dataset to sample from.

    Initial Idea:
        A `MultimodalSampler` can coordinate indices across modality datasets to ensure
        balanced or paired sampling before passing to `BaseDataLoader`.
        Note: This is optional and meant as a design hint, not a constraint.
    """
