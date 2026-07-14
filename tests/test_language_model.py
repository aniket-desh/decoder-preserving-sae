from types import SimpleNamespace

import torch

from dpsae.language_model import (
    ActivationStats,
    _replace_block_output,
    _transformer_blocks,
    answer_logit_difference,
    estimate_activation_stats,
)


def test_transformer_blocks_supports_gpt2_and_gpt_neox_layouts():
    gpt2_blocks = [object(), object()]
    neox_blocks = [object(), object(), object()]
    assert _transformer_blocks(
        SimpleNamespace(transformer=SimpleNamespace(h=gpt2_blocks))
    ) is gpt2_blocks
    assert _transformer_blocks(
        SimpleNamespace(gpt_neox=SimpleNamespace(layers=neox_blocks))
    ) is neox_blocks


def test_activation_stats_round_trip():
    x = torch.randn(32, 7) * 3 + 4
    stats = estimate_activation_stats(x)
    assert torch.allclose(stats.denormalize(stats.normalize(x)), x, atol=1e-5)
    restored = ActivationStats.from_state_dict(stats.state_dict(), torch.device("cpu"))
    assert torch.allclose(restored.mean, stats.mean)


def test_block_tuple_replacement_preserves_tail():
    original = (torch.zeros(2, 3), "cache")
    replacement = torch.ones(2, 3)
    assert _replace_block_output(original, replacement) == (replacement, "cache")


def test_answer_logit_difference_uses_last_unpadded_position():
    logits = torch.zeros(2, 4, 10)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    logits[0, 2, 3], logits[0, 2, 4] = 5, 2
    logits[1, 3, 6], logits[1, 3, 7] = 1, 4
    result = answer_logit_difference(
        logits, mask, torch.tensor([3, 6]), torch.tensor([4, 7])
    )
    assert torch.equal(result, torch.tensor([3.0, -3.0]))
