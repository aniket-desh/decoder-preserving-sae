from pathlib import Path

import numpy as np
import torch

from dpsae.corpus import MemmapTokenBatcher, TokenRange


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
