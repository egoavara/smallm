"""데이터 로딩."""

import torch
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader, IterableDataset
from typing import Iterator, TYPE_CHECKING
from tqdm import tqdm

if TYPE_CHECKING:
    from ..tokenizer.base import Tokenizer

SHUFFLE_SEED = 42
SHUFFLE_BUFFER = 10000


CHATML_USER = "<|im_start|>user\n{content}<|im_end|>\n"
CHATML_ASSISTANT = "<|im_start|>assistant\n{content}<|im_end|>\n"
CHATML_SYSTEM = "<|im_start|>system\n{content}<|im_end|>\n"


def format_chatml(messages: list[dict], system: str = "") -> str:
    result = ""
    if system:
        result += CHATML_SYSTEM.format(content=system)

    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role in ("user", "human", "prompter"):
            result += CHATML_USER.format(content=content)
        elif role in ("assistant", "gpt", "bot"):
            result += CHATML_ASSISTANT.format(content=content)

    return result


def _load_wikitext(split: str = "train"):
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split, streaming=True)
    ds = ds.filter(lambda x: x["text"] and x["text"].strip())
    return ds.map(lambda x: {"text": x["text"]}, remove_columns=ds.column_names)


def _load_tinystories(split: str = "train"):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    ds = ds.filter(lambda x: x["text"] and x["text"].strip())
    return ds.map(lambda x: {"text": x["text"]}, remove_columns=ds.column_names)


def _load_openwebtext(split: str = "train"):
    ds = load_dataset("Skylion007/openwebtext", split=split, streaming=True)
    ds = ds.filter(lambda x: x["text"] and x["text"].strip())
    return ds.map(lambda x: {"text": x["text"]}, remove_columns=ds.column_names)


def _load_openassistant(split: str = "train"):
    ds = load_dataset("OpenAssistant/oasst1", split=split)

    messages_by_id = {}
    children_by_parent = {}
    roots = []

    for item in ds:
        # 영어만 필터링
        if item.get("lang") != "en":
            continue

        msg_id = item["message_id"]
        parent_id = item["parent_id"]
        role = "user" if item["role"] == "prompter" else "assistant"

        messages_by_id[msg_id] = {
            "message_id": msg_id,
            "parent_id": parent_id,
            "role": role,
            "content": item["text"],
            "lang": item.get("lang", "en"),
        }

        if parent_id is None:
            roots.append(msg_id)
        else:
            if parent_id not in children_by_parent:
                children_by_parent[parent_id] = []
            children_by_parent[parent_id].append(msg_id)

    conversations = []

    def dfs(msg_id: str, path: list[dict]):
        msg = messages_by_id[msg_id]
        new_path = path + [{"role": msg["role"], "content": msg["content"]}]
        children = children_by_parent.get(msg_id, [])
        if not children:
            if len(new_path) >= 2:
                conversations.append(format_chatml(new_path))
        else:
            for child_id in children:
                dfs(child_id, new_path)

    for root_id in roots:
        dfs(root_id, [])

    from datasets import Dataset
    return Dataset.from_dict({"text": conversations}).to_iterable_dataset()


def _load_alpaca(split: str = "train"):
    ds = load_dataset("tatsu-lab/alpaca", split=split, streaming=True)

    def to_chatml(item):
        content = item["instruction"]
        if item["input"]:
            content += "\n\n" + item["input"]
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": item["output"]},
        ]
        return {"text": format_chatml(messages)}

    return ds.map(to_chatml, remove_columns=ds.column_names)


def _load_dolly(split: str = "train"):
    ds = load_dataset("databricks/databricks-dolly-15k", split=split, streaming=True)

    def to_chatml(item):
        content = item["instruction"]
        if item["context"]:
            content += "\n\n" + item["context"]
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": item["response"]},
        ]
        return {"text": format_chatml(messages)}

    return ds.map(to_chatml, remove_columns=ds.column_names)


def _load_ultrachat(split: str = "train"):
    ds = load_dataset("stingning/ultrachat", split=split, streaming=True)

    def to_chatml(item):
        data = item.get("data", [])
        if len(data) < 2:
            return {"text": ""}
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": c}
            for i, c in enumerate(data)
        ]
        return {"text": format_chatml(messages)}

    ds = ds.map(to_chatml, remove_columns=ds.column_names)
    return ds.filter(lambda x: x["text"].strip())


LOADERS = {
    "wikitext": _load_wikitext,
    "tinystories": _load_tinystories,
    "openwebtext": _load_openwebtext,
    "openassistant": _load_openassistant,
    "alpaca": _load_alpaca,
    "dolly": _load_dolly,
    "ultrachat": _load_ultrachat,
}


def _load_datasets(
    datasets: list[str] | tuple[str, ...],
    split: str = "train",
    shuffle: bool = True,
):
    """여러 데이터셋을 로드하고 interleave + shuffle."""
    ds_list = []
    for ds_name in datasets:
        if ds_name not in LOADERS:
            raise ValueError(f"Unknown dataset: {ds_name}. Available: {list(LOADERS.keys())}")
        ds_list.append(LOADERS[ds_name](split))

    if len(ds_list) == 1:
        combined = ds_list[0]
    else:
        combined = interleave_datasets(ds_list)

    if shuffle:
        combined = combined.shuffle(seed=SHUFFLE_SEED, buffer_size=SHUFFLE_BUFFER)

    return combined


def iter_texts(
    datasets: list[str] | tuple[str, ...],
    split: str = "train",
    sample_ratio: float = 1.0,
    shuffle: bool = True,
    show_progress: bool = True,
) -> Iterator[str]:
    import random

    combined = _load_datasets(datasets, split, shuffle)

    iterator = iter(combined)
    if show_progress:
        iterator = tqdm(iterator, desc="Loading")

    for item in iterator:
        if sample_ratio < 1.0 and random.random() > sample_ratio:
            continue
        yield item["text"]


class TokenDataset(IterableDataset):
    def __init__(
        self,
        datasets: list[str] | tuple[str, ...],
        tokenizer: "Tokenizer",
        seq_len: int,
        split: str = "train",
        shuffle: bool = True,
    ):
        self.datasets = list(datasets)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.shuffle = shuffle

    def __iter__(self):
        buffer = []
        combined = _load_datasets(self.datasets, self.split, self.shuffle)

        for item in combined:
            tokens = self.tokenizer.encode(item["text"])
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                yield (
                    torch.tensor(chunk[:-1], dtype=torch.long),
                    torch.tensor(chunk[1:], dtype=torch.long),
                )


def create_dataloader(
    datasets: list[str] | tuple[str, ...],
    tokenizer: "Tokenizer",
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = TokenDataset(datasets, tokenizer, seq_len, shuffle=shuffle)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def list_datasets() -> list[str]:
    return list(LOADERS.keys())
