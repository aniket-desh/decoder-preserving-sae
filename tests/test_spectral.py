import pytest
import torch

from dpsae.decoder_distance import decoder_distance
from dpsae.spectral import (
    decoder_gains,
    optimal_decoder_tail,
    structured_decoder_scores,
    truncated_svd,
)


def test_truncated_svd_attains_predicted_decoder_tail() -> None:
    torch.manual_seed(0)
    n, d, rank = 18, 12, 8
    u, _ = torch.linalg.qr(torch.randn(n, rank, dtype=torch.float64))
    v, _ = torch.linalg.qr(torch.randn(d, rank, dtype=torch.float64))
    singular_values = torch.logspace(1, -1, rank, dtype=torch.float64)
    x = (u * singular_values) @ v.mT
    tau = 1.7
    ridge = tau / n
    predicted = optimal_decoder_tail(singular_values, tau)

    for retained_rank in range(rank + 1):
        x_rank = truncated_svd(x, retained_rank)
        observed = decoder_distance(x, x_rank, ridge=ridge, reduction="sum")
        assert observed.item() == pytest.approx(
            predicted[retained_rank].item(), rel=1e-9, abs=1e-10
        )


def test_decoder_gain_is_monotone_and_bounded() -> None:
    singular_values = torch.tensor([0.1, 1.0, 10.0])
    gains = decoder_gains(singular_values, tau=1.0)
    assert torch.all(gains[1:] > gains[:-1])
    assert torch.all((gains > 0) & (gains < 1))


def test_structured_prior_crosses_predicted_selection_boundary() -> None:
    singular_values = torch.tensor([4.0, 1.0], dtype=torch.float64)
    gains = decoder_gains(singular_values, tau=1.0)
    crossover = (gains[0] / gains[1]).square()
    below = structured_decoder_scores(
        singular_values, torch.tensor([1.0, 0.9 * crossover]), tau=1.0
    )
    above = structured_decoder_scores(
        singular_values, torch.tensor([1.0, 1.1 * crossover]), tau=1.0
    )
    assert below.argmax().item() == 0
    assert above.argmax().item() == 1
