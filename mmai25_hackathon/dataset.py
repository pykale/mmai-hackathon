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

from .load_data.cxr import load_chest_xray_image, load_mimic_cxr_metadata
from .load_data.ecg import load_ecg_record, load_mimic_iv_ecg_record_list
from .load_data.echo import load_echo_dicom, load_mimic_iv_echo_record_list

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
        self.records = load_mimic_cxr_metadata(args.data_path)
        self.subject_ids = self.records["subject_id"].tolist()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, sample_ID: int):
        record_idx = self.records[self.records.subject_id == sample_ID]
        samples = []
        for idx in record_idx:
            path = idx["cxr_path"]
            image = load_chest_xray_image(path)
            item = {"image": image, "subject_id": record_idx["subject_id"]}
            samples.append(item)
        return samples

    def modality(self) -> str:
        """Return the modality of the dataset."""
        return "CXR"


class ECGDataset(BaseDataset):
    """Example subclass for an ECG dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Loading the ECG data (records contains patient id, hea path) in df frame
        self.records = load_mimic_iv_ecg_record_list(args.data_path)
        self.subject_ids = self.records["subject_id"].tolist()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, sample_ID: int):
        """Return samples for one sampleID from the dataset."""
        # record_idx = self.records[idx]
        # signals, fields = load_ecg_record(record_idx["hea_path"])
        # return {"signals": signals, "fields": fields, "subject_id": record_idx["subject_id"]}
        record_idx = self.records[self.records.subject_id == sample_ID]
        samples = []
        for idx in record_idx:
            signals, fields = load_ecg_record(idx["hea_path"])
            item = {"signals": signals, "fields": fields, "subject_id": record_idx["subject_id"]}
            samples.append(item)
        return samples

    def __repr__(self) -> str:
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self) -> str:
        """Return any extra information about the dataset."""
        return f"sample_size={len(self.subject_ids)}"

    def modality(self) -> str:
        """Return the modality of the dataset."""
        return "ECG"

    def __add__(self, other):
        """
        Combine with another dataset. Assume other is a single sample.

        Override in subclasses to implement multimodal aggregation.

        Args:
            other: Another dataset to combine with this one.

        Initial Idea:
            Use `__add__` to align and merge heterogeneous modalities into a single
            dataset, keeping shared IDs synchronized.
            Note: This is not mandatory; treat it as a sketch you can refine or replace.
        """
        self.records = self.records.merge(other.records, on="subject_id", suffixes=("", "_other"), how="outer")
        self.subject_ids = self.records["subject_id"].tolist()

        # TODO: takes a single sample from other, find corresponding sample in this dataset?
        # i.e. find idx where sample_id matches from other and call get_item on all of those indices?

        return self


class EchoDataset(BaseDataset):
    """Example subclass for an ECHO dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = load_mimic_iv_echo_record_list(args.data_path)
        self.subject_ids = self.records["subject_id"].tolist()

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.records)

    def __getitem__(self, sample_ID: int):
        """Return a single sample from the dataset."""
        # record = self.records.iloc[idx]
        # sample_path = record["echo_path"]
        # frames, meta = load_echo_dicom(sample_path)
        # return {"frames": frames, "metadata": meta, "subject_id": record["subject_id"]}
        record_idx = self.records[self.records.subject_id == sample_ID]
        samples = []
        for idx in record_idx:
            sample_path = idx["echo_path"]
            frames, meta = load_echo_dicom(sample_path)
            item = {"frames": frames, "metadata": meta, "subject_id": idx["subject_id"]}
            samples.append(item)
        return samples

    def extra_repr(self) -> str:
        """Return any extra information about the dataset."""
        return f"sample_size={len(self)}, subjects={len(set(self.subject_ids))}"

    def modality(self) -> str:
        """Return the modality of the dataset."""
        return "echo"

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
        self.records = self.records.merge(other.records, on="subject_id", suffixes=("", "_other"), how="outer")
        self.subject_ids = self.records["subject_id"].tolist()
        return self


class MultimodalDataset(BaseDataset):
    """Example subclass for a multimodal dataset."""

    def __init__(self, datasets: list[BaseDataset], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datasets = datasets
        # _dataset = datasets[0]
        # if not isinstance(_dataset, BaseDataset):
        #    raise ValueError("All elements in datasets must be instances of BaseDataset.")
        # if len(datasets) > 1:
        #    for ds in datasets[1:]:
        #        if not isinstance(ds, BaseDataset):
        #            raise ValueError("All elements in datasets must be instances of BaseDataset.")
        #        _dataset.__add__(ds)
        # self.dataset = _dataset

        # get union of all subject IDs in each dataset
        self.subject_ids = list(set().union(*(ds.subject_ids for ds in datasets)))
        print(f"MultimodalDataset initialized with {len(self.subject_ids)} unique subjects.")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """Return a single sample from the dataset."""
        subject_ID = self.subject_ids[idx]
        results = {}
        for dataset in self.datasets:
            items = dataset.__getitem__(subject_ID)
            results[dataset.modality()] = items
        return results

        # get dictionaries for each dataset
        # results = {}
        # for ds in self.datasets:
        #    dict_result = ds.__getitem__(idx) # TODO: assumes idx is same for each sample. replace idx with sample ID
        #    results[ds.modality()] = dict_result

        # or primary dataset and __add__ in others
        # primary_ds = self.datasets[0]
        # item = primary_ds.__getitem__(idx)
        # for ds in self.datasets[1:]:
        #    items = ds.__add__(item)
        # return results

    def extra_repr(self) -> str:
        """Return any extra information about the dataset."""
        return self.dataset.extra_repr()


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

    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: list = None,
        exclude_keys: list = None,
        **kwargs,
    ):
        super().__init__(dataset, batch_size, shuffle, follow_batch, exclude_keys, **kwargs)

        # collate_fn=lambda data_list: Batch.from_data_list(
        #    data_list, follow_batch),


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
