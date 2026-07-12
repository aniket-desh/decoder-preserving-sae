"""Deterministic natural-text token storage for language SAE experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def prepare_token_memmap(
    output_path: Path,
    *,
    tokenizer,
    token_count: int,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str = "text",
    document_batch: int = 256,
) -> dict:
    """Stream and tokenize documents into a fixed-size uint16 memmap."""

    from datasets import load_dataset

    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("tokenizer vocabulary does not fit in uint16")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("token_count") == token_count:
            return metadata

    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    tokens = np.memmap(output_path, mode="w+", dtype=np.uint16, shape=(token_count,))
    written = 0
    documents: list[str] = []

    def flush(batch: list[str]) -> None:
        nonlocal written
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for document_tokens in encoded:
            if written >= token_count:
                break
            sequence = document_tokens + [tokenizer.eos_token_id]
            take = min(len(sequence), token_count - written)
            tokens[written : written + take] = np.asarray(sequence[:take], dtype=np.uint16)
            written += take

    for row in dataset:
        text = row.get(text_column)
        if not text:
            continue
        documents.append(text)
        if len(documents) >= document_batch:
            flush(documents)
            documents.clear()
        if written >= token_count:
            break
    if documents and written < token_count:
        flush(documents)
    tokens.flush()
    if written != token_count:
        raise RuntimeError(f"dataset ended after {written:,} of {token_count:,} requested tokens")
    metadata = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "token_count": token_count,
        "dtype": "uint16",
        "tokenizer": tokenizer.name_or_path,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


@dataclass(frozen=True)
class TokenRange:
    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


class MemmapTokenBatcher:
    """Draw reproducible contiguous sequences from a token range."""

    def __init__(
        self,
        path: Path,
        *,
        token_count: int,
        token_range: TokenRange,
        sequence_length: int,
        batch_size: int,
        seed: int,
    ) -> None:
        if token_range.stop > token_count or token_range.size <= sequence_length:
            raise ValueError("invalid token range")
        self.tokens = np.memmap(path, mode="r", dtype=np.uint16, shape=(token_count,))
        self.token_range = token_range
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(seed)

    def batch(self) -> Tensor:
        high = self.token_range.stop - self.sequence_length
        starts = torch.randint(
            self.token_range.start,
            high,
            (self.batch_size,),
            generator=self.generator,
        ).tolist()
        array = np.stack(
            [self.tokens[start : start + self.sequence_length] for start in starts]
        ).astype(np.int64, copy=False)
        return torch.from_numpy(array)
