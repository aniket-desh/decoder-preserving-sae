from pathlib import Path

import pytest
import torch

from experiments import exp05_semantic_recurrence as review


class FakeTokenizer:
    name_or_path = "fake"

    def decode(self, token_ids, **_kwargs):
        return " ".join(str(token_id) for token_id in token_ids)


def record(token: str, position: int) -> dict:
    return {
        "token_id": position,
        "surface": f" {token}",
        "normalized": token,
        "previous_normalized": "previous",
        "next_normalized": "next",
        "position": position,
        "absolute_offset": position,
        "context": f"context {token}",
    }


def test_familywise_alignment_detects_a_perfect_lexical_partition():
    records = [record("cat" if index < 16 else "dog", index) for index in range(128)]
    vector = review._unit_feature(set(range(16)), 128)
    protocol = review.ReviewProtocol(permutation_samples=64)

    alignments = review.mode_alignments(
        vector,
        records,
        mode_id="mode",
        protocol=protocol,
    )
    by_feature = {row["feature"]: row for row in alignments}

    assert by_feature["token:cat"]["absolute_cosine"] == pytest.approx(1)
    assert by_feature["token:cat"]["qualifies"] is True


def test_recurrence_records_never_cross_the_declared_range():
    cache = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
        "starts": torch.tensor([10, 20, 30]),
    }

    records = review._recurrence_records(
        FakeTokenizer(),
        cache,
        token_range=(20, 34),
        radius=1,
    )

    assert len(records) == 8
    assert {row["absolute_offset"] for row in records} == set(range(20, 24)) | set(
        range(30, 34)
    )


def test_review_table_contains_every_mode(tmp_path: Path):
    mode = {
        "mode_id": "m0",
        "seed": 0,
        "group_position": 2,
        "side": "top",
        "rank": 1,
        "eigenvalue": 0.1,
        "row_shuffle_absolute_ratio": 0.5,
        "top_alignments": [],
        "recurring_features": [],
        "positive_extremes": [],
        "negative_extremes": [],
        "automated_verdict": "no_cross_seed_group_lexical_recurrence",
    }
    path = tmp_path / "review.tsv"
    review.write_review_table(path, {"modes": [mode]})

    assert path.read_text().count("\n") == 2
    assert "m0" in path.read_text()
