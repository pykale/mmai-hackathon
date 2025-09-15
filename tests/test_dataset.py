"""Tests for base dataset/dataloader/sampler utilities.

This suite validates the public classes in ``mmai25_hackathon.dataset``:

- ``BaseDataset``: abstract interface; ``__len__``/``__getitem__`` must be implemented; ``__repr__``
  uses ``extra_repr``; ``prepare_data`` is optional.
- ``BaseDataLoader``: batches simple PyG graphs and yields ``Batch`` objects.
- ``BaseSampler``: remains abstract for iteration (``__iter__`` raises ``NotImplementedError``).

Prerequisite
------------
Graph batching test requires ``torch`` and ``torch_geometric``; if unavailable, the test is skipped.
"""

import pytest

from mmai25_hackathon.dataset import BaseDataLoader, BaseDataset, BaseSampler


def test_base_dataset_instantiation_raises() -> None:
    with pytest.raises(NotImplementedError):
        BaseDataset()


def test_incomplete_subclass_abstract_methods_raise() -> None:
    class Incomplete(BaseDataset):
        def __init__(self) -> None:
            # Override to avoid BaseDataset.__init__ raising, but keep abstract methods unimplemented
            pass

    ds = Incomplete()
    with pytest.raises(NotImplementedError):
        _ = len(ds)
    with pytest.raises(NotImplementedError):
        _ = ds[0]
    with pytest.raises(NotImplementedError):
        _ = ds + ds
    with pytest.raises(NotImplementedError):
        ds.prepare_data()


def test_complete_subclass_repr_and_len() -> None:
    class ToyDataset(BaseDataset):
        def __init__(self, items: list[int]) -> None:
            self._items = list(items)

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int) -> int:
            return self._items[idx]

    ds = ToyDataset([1, 2, 3])
    assert len(ds) == 3, f"Expected length 3, got {len(ds)}"
    assert repr(ds) == "ToyDataset(sample_size=3)", f"Unexpected repr: {repr(ds)!r}"


def test_base_dataloader_batches_graphs() -> None:
    # Skip if torch_geometric/torch unavailable
    pytest.importorskip("torch_geometric")
    torch = pytest.importorskip("torch")
    from torch_geometric.data import Batch, Data

    class GraphDataset:
        def __init__(self) -> None:
            self._graphs = [
                Data(x=torch.randn(3, 2), edge_index=torch.empty(2, 0, dtype=torch.long)),
                Data(x=torch.randn(2, 2), edge_index=torch.empty(2, 0, dtype=torch.long)),
                Data(x=torch.randn(4, 2), edge_index=torch.empty(2, 0, dtype=torch.long)),
                Data(x=torch.randn(1, 2), edge_index=torch.empty(2, 0, dtype=torch.long)),
                Data(x=torch.randn(2, 2), edge_index=torch.empty(2, 0, dtype=torch.long)),
            ]

        def __len__(self) -> int:
            return len(self._graphs)

        def __getitem__(self, idx: int) -> Data:  # type: ignore[name-defined]
            return self._graphs[idx]

    ds = GraphDataset()  # type: ignore[arg-type]
    loader = BaseDataLoader(ds, batch_size=2, shuffle=False)  # type: ignore[arg-type]

    total_graphs = 0
    for batch in loader:
        assert isinstance(batch, Batch), f"Loader should return Batch, got {type(batch)!r}"
        assert getattr(batch, "num_graphs", 0) > 0, "Batch missing num_graphs or is zero"
        total_graphs += int(batch.num_graphs)

    assert total_graphs == len(ds), f"Total graphs {total_graphs} != dataset size {len(ds)}"


def test_base_sampler_iteration_not_implemented() -> None:
    # BaseSampler inherits torch.utils.data.Sampler: __iter__ is abstract and should raise
    s = BaseSampler(data_source=range(5))  # type: ignore[call-arg]
    with pytest.raises(NotImplementedError):
        _ = next(iter(s))
