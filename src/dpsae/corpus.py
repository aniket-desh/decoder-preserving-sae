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
    token_offset: int = 0,
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str = "text",
    document_batch: int = 256,
) -> dict:
    """Stream a fixed token slice into a uint16 memmap.

    ``token_offset`` counts the same document tokens and inserted EOS markers
    as the output stream, so a later immutable shard can be generated without
    overlapping an earlier corpus.
    """

    from datasets import load_dataset

    if token_offset < 0:
        raise ValueError("token_offset must be nonnegative")
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError("tokenizer vocabulary does not fit in uint16")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata.get("token_count") == token_count
            and metadata.get("token_offset", 0) == token_offset
        ):
            return metadata

    dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    partial_path = output_path.with_suffix(output_path.suffix + ".partial")
    partial_path.unlink(missing_ok=True)
    tokens = np.memmap(partial_path, mode="w+", dtype=np.uint16, shape=(token_count,))
    streamed = 0
    written = 0
    documents: list[str] = []

    def flush(batch: list[str]) -> None:
        nonlocal streamed, written
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for document_tokens in encoded:
            if written >= token_count:
                break
            sequence = document_tokens + [tokenizer.eos_token_id]
            sequence_stop = streamed + len(sequence)
            overlap_start = max(streamed, token_offset)
            overlap_stop = min(sequence_stop, token_offset + token_count)
            if overlap_start < overlap_stop:
                source_start = overlap_start - streamed
                take = overlap_stop - overlap_start
                tokens[written : written + take] = np.asarray(
                    sequence[source_start : source_start + take], dtype=np.uint16
                )
                written += take
            streamed = sequence_stop

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
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"dataset ended after {written:,} of {token_count:,} requested tokens")
    metadata = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "token_count": token_count,
        "token_offset": token_offset,
        "dtype": "uint16",
        "tokenizer": tokenizer.name_or_path,
    }
    partial_path.replace(output_path)
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_metadata.replace(metadata_path)
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

    def batch_with_starts(self) -> tuple[Tensor, Tensor]:
        """Draw a batch and return the absolute start offset of every sequence."""

        high = self.token_range.stop - self.sequence_length
        starts = torch.randint(
            self.token_range.start,
            high,
            (self.batch_size,),
            generator=self.generator,
        )
        array = np.stack(
            [
                self.tokens[int(start) : int(start) + self.sequence_length]
                for start in starts
            ]
        ).astype(np.int64, copy=False)
        return torch.from_numpy(array), starts

    def batch(self) -> Tensor:
        return self.batch_with_starts()[0]

    def load_generator_state(self, state: Tensor) -> None:
        """Restore a saved CPU generator state after device-mapped checkpoint loading."""

        self.generator.set_state(state.detach().cpu())
