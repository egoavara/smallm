"""Dataset utilities."""

from .dataset import (
    TextDataset,
    load_wikitext,
    load_dataset_by_name,
    create_dataloader,
    InfiniteDataLoader,
)
from .mixed import MixedTextDataset, load_mixed_dataset
from .streaming import (
    StreamingTextDataset,
    StreamingMixedDataset,
    load_streaming_dataset,
    load_streaming_mixed_dataset,
    create_streaming_dataloader,
)
from .registry import list_datasets, get_dataset_info, DATASET_INFO

# 로더들 import하여 자동 등록
from . import loaders  # noqa: F401

__all__ = [
    # Dataset classes (in-memory)
    "TextDataset",
    "MixedTextDataset",
    "InfiniteDataLoader",
    # Dataset classes (streaming)
    "StreamingTextDataset",
    "StreamingMixedDataset",
    # Loader functions (in-memory)
    "load_wikitext",
    "load_dataset_by_name",
    "load_mixed_dataset",
    "create_dataloader",
    # Loader functions (streaming)
    "load_streaming_dataset",
    "load_streaming_mixed_dataset",
    "create_streaming_dataloader",
    # Registry
    "list_datasets",
    "get_dataset_info",
    "DATASET_INFO",
]
