import pytest
import torch

from dpsae.exp04b_execution import (
    confirmatory_example_splits,
    duplicate_state_ranking,
)


def test_confirmatory_split_exhausts_discovery_and_reserves_validation():
    examples = {
        "discovery": list(range(8)),
        "validation": [8, 9],
    }
    split = confirmatory_example_splits(
        examples,
        ranking_examples=6,
        selection_examples=2,
    )

    assert split == {"ranking": list(range(6)), "selection": [6, 7], "test": [8, 9]}
    with pytest.raises(ValueError, match="exhaust discovery"):
        confirmatory_example_splits(examples, ranking_examples=5, selection_examples=2)


def test_duplicate_state_ranking_returns_frequency_matched_control():
    positive = torch.zeros(16, 6)
    negative = torch.zeros(16, 6)
    positive[:, 0] = torch.arange(16) % 2
    negative[:, 0] = 0
    positive[:, 2] = 1
    negative[:, 2] = 0
    positive[:, 1] = (torch.arange(16) % 4 == 0).float()
    negative[:, 1] = (torch.arange(16) % 4 == 0).float()
    positive[:, 3] = torch.arange(16) % 2
    negative[:, 3] = torch.arange(16) % 2

    ranking, random = duplicate_state_ranking((positive, negative), maximum=2)

    assert set(ranking[:2].tolist()) == {0, 2}
    assert set(random.tolist()) == {1, 3}
