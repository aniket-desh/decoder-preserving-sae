from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from dpsae.corpus import MemmapTokenBatcher, TokenRange, prepare_token_memmap


def test_prepare_token_memmap_writes_nonoverlapping_stream_slice(
    tmp_path: Path, monkeypatch
):
    rows = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: iter(rows)),
    )

    class Tokenizer:
        vocab_size = 100
        eos_token_id = 99
        name_or_path = "fake"

        def __call__(self, texts, *, add_special_tokens):
            assert not add_special_tokens
            mapping = {"a": [1, 2, 3], "b": [4, 5], "c": [6, 7, 8]}
            return {"input_ids": [mapping[text] for text in texts]}

    path = tmp_path / "tail.bin"
    metadata = prepare_token_memmap(
        path,
        tokenizer=Tokenizer(),
        token_count=5,
        token_offset=4,
        dataset_name="fake",
        dataset_config=None,
        split="train",
        document_batch=2,
    )

    values = np.memmap(path, mode="r", dtype=np.uint16, shape=(5,))
    assert values.tolist() == [4, 5, 99, 6, 7]
    assert metadata["token_offset"] == 4
    assert not path.with_suffix(".bin.partial").exists()


def test_memmap_token_batcher_is_reproducible(tmp_path: Path):
    path = tmp_path / "tokens.bin"
    values = np.memmap(path, mode="w+", dtype=np.uint16, shape=(100,))
    values[:] = np.arange(100)
    values.flush()
    kwargs = dict(
        token_count=100,
        token_range=TokenRange(10, 90),
        sequence_length=8,
        batch_size=4,
        seed=3,
    )
    first = MemmapTokenBatcher(path, **kwargs).batch()
    second = MemmapTokenBatcher(path, **kwargs).batch()
    assert torch.equal(first, second)
    assert first.shape == (4, 8)
    assert torch.all(first[:, 1:] - first[:, :-1] == 1)


def test_memmap_token_batcher_restores_device_mapped_generator_state(tmp_path: Path):
    path = tmp_path / "tokens.bin"
    values = np.memmap(path, mode="w+", dtype=np.uint16, shape=(100,))
    values[:] = np.arange(100)
    values.flush()
    kwargs = dict(
        token_count=100,
        token_range=TokenRange(10, 90),
        sequence_length=8,
        batch_size=4,
        seed=3,
    )
    original = MemmapTokenBatcher(path, **kwargs)
    original.batch()
    state = original.generator.get_state()
    expected = original.batch()
    if torch.cuda.is_available():
        state = state.cuda()

    restored = MemmapTokenBatcher(path, **kwargs)
    restored.load_generator_state(state)

    assert torch.equal(restored.batch(), expected)


def test_memmap_token_batcher_reports_reproducible_absolute_starts(tmp_path: Path):
    path = tmp_path / "tokens.bin"
    values = np.memmap(path, mode="w+", dtype=np.uint16, shape=(100,))
    values[:] = np.arange(100)
    values.flush()
    kwargs = dict(
        token_count=100,
        token_range=TokenRange(10, 90),
        sequence_length=8,
        batch_size=4,
        seed=5,
    )

    batch, starts = MemmapTokenBatcher(path, **kwargs).batch_with_starts()

    assert starts.shape == (4,)
    assert torch.equal(batch[:, 0], starts)
    assert int(starts.min()) >= 10
    assert int(starts.max()) < 82
