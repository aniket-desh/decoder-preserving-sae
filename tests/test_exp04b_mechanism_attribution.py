from argparse import Namespace

import pytest
import torch

from dpsae.language_sae import BatchTopKSAE
from experiments import exp04b_mechanism_attribution as mechanism


def test_tangent_identity_and_eigenbasis_split_are_exact():
    generator = torch.Generator().manual_seed(17)
    original = torch.randn(5, 7, generator=generator)
    reconstructed = original + 0.08 * torch.randn(
        original.shape, generator=generator
    )

    row = mechanism.tangent_group_statistics(original, reconstructed, ridge=0.3)

    assert row["decomposition_residual"] < 2e-5
    assert row["exact_numerator"] == pytest.approx(
        row["tangent_numerator"]
        + row["tangent_remainder_cross"]
        + row["remainder_numerator"],
        rel=2e-5,
        abs=2e-7,
    )
    assert row["tangent_numerator"] == pytest.approx(
        row["tangent_diagonal_numerator"]
        + row["tangent_off_diagonal_numerator"],
        rel=2e-5,
        abs=2e-7,
    )
    assert -1 <= row["tangent_exact_cosine"] <= 1


def test_orthogonal_residual_gram_is_psd_and_matches_explicit_construction():
    generator = torch.Generator().manual_seed(23)
    code = torch.relu(torch.randn(4, 5, generator=generator))
    decoder = torch.nn.functional.normalize(
        torch.randn(5, 3, generator=generator), dim=1
    )
    bias = torch.randn(3, generator=generator)

    actual = mechanism.orthogonal_residual_gram(code, decoder, bias)
    beta = bias.norm()
    atom_bias = decoder @ bias
    residual = (1 - atom_bias.square() / beta.square()).clamp_min(0).sqrt()
    hypothetical_decoder = torch.zeros(5, 6)
    hypothetical_decoder[:, 0] = atom_bias / beta
    hypothetical_decoder[torch.arange(5), torch.arange(5) + 1] = residual
    hypothetical_bias = torch.zeros(6)
    hypothetical_bias[0] = beta
    hypothetical_reconstruction = code @ hypothetical_decoder + hypothetical_bias
    expected = hypothetical_reconstruction @ hypothetical_reconstruction.mT

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.linalg.eigvalsh(actual).min() >= -1e-5
    torch.testing.assert_close(
        hypothetical_decoder.square().sum(1), torch.ones(5), atol=1e-6, rtol=1e-5
    )
    torch.testing.assert_close(
        hypothetical_decoder @ hypothetical_bias, decoder @ bias, atol=1e-6, rtol=1e-5
    )


def test_nonorthogonal_counterfactual_is_inactive_for_orthogonal_zero_bias_decoder():
    original = torch.diag(torch.tensor([2.0, 3.0]))
    code = torch.eye(2)
    decoder = original.clone()
    bias = torch.zeros(2)

    row = mechanism.nonorthogonal_group_statistics(
        original, code, decoder, bias, ridge=0.2
    )

    assert row["exact_numerator"] == pytest.approx(0)
    assert row["orthogonal_numerator"] == pytest.approx(0)
    assert row["nonorthogonality_benefit"] == pytest.approx(0)
    assert row["orthogonal_gram_min_eigenvalue"] >= 0


def _tangent_row(
    *,
    exact: float,
    tangent: float,
    diagonal: float,
    remainder_ratio: float = 0.0,
    cosine: float = 1.0,
) -> dict[str, float]:
    return {
        "exact_numerator": exact,
        "denominator": 1.0,
        "tangent_numerator": tangent,
        "tangent_diagonal_numerator": diagonal,
        "tangent_off_diagonal_numerator": tangent - diagonal,
        "remainder_ratio": remainder_ratio,
        "tangent_exact_cosine": cosine,
        "gram_error_numerator": exact,
        "gram_reference": 1.0,
    }


def test_tangent_pass_fail_metrics_distinguish_linear_and_endpoint_mechanisms():
    sufficient = mechanism.tangent_pair_report(
        [_tangent_row(exact=2.0, tangent=2.0, diagonal=1.5)],
        [_tangent_row(exact=1.0, tangent=1.0, diagonal=0.5)],
        bootstrap_samples=0,
        seed=1,
    )
    endpoint = mechanism.tangent_pair_report(
        [_tangent_row(exact=2.0, tangent=0.5, diagonal=0.25, remainder_ratio=1.0)],
        [_tangent_row(exact=1.0, tangent=1.0, diagonal=0.5, remainder_ratio=1.0)],
        bootstrap_samples=0,
        seed=1,
    )

    assert sufficient["diagnostics"]["exact_favors_dpsae"]
    assert sufficient["diagnostics"]["source_tangent_sufficient"]
    assert not sufficient["diagnostics"]["endpoint_nonlinearity_necessary"]
    assert endpoint["diagnostics"]["endpoint_nonlinearity_necessary"]
    assert not endpoint["diagnostics"]["source_tangent_sufficient"]


def _nonorth_row(*, exact: float, orthogonal: float) -> dict[str, float]:
    return {
        "exact_numerator": exact,
        "orthogonal_numerator": orthogonal,
        "nonorthogonality_benefit": orthogonal - exact,
        "denominator": 1.0,
    }


def test_nonorthogonal_pass_fail_metrics_detect_necessary_counterfactual():
    report = mechanism.nonorthogonal_pair_report(
        [_nonorth_row(exact=2.0, orthogonal=1.0)],
        [_nonorth_row(exact=1.0, orthogonal=2.0)],
        bootstrap_samples=0,
        seed=3,
    )

    assert report["diagnostics"]["exact_favors_dpsae"]
    assert report["diagnostics"]["nonorthogonality_necessary"]
    assert report["nonorthogonality_contrast"]["estimate"] > 0


def test_tangent_stage_streams_groups_and_records_input_hashes(tmp_path):
    generator = torch.Generator().manual_seed(31)
    activations = torch.randn(2, 4, 3, generator=generator)
    cache = tmp_path / "natural.pt"
    torch.save({"activations": activations.half()}, cache)
    reconstruction_dir = tmp_path / "reconstructions"
    reconstruction_dir.mkdir()
    mse = activations + 0.2 * torch.randn(activations.shape, generator=generator)
    dpsae = activations + 0.05 * torch.randn(activations.shape, generator=generator)
    torch.save(mse.half(), reconstruction_dir / "mse_s0.pt")
    torch.save(dpsae.half(), reconstruction_dir / "dpsae_s0.pt")
    output = tmp_path / "tangent.json"
    args = Namespace(
        natural_cache=cache,
        static_calibration=None,
        ridge=0.2,
        group_size=4,
        exact_tokens=8,
        max_groups=1,
        bootstrap_samples=32,
        seed=4,
        pair=None,
        reconstruction_dir=reconstruction_dir,
        output=output,
    )

    result = mechanism.run_tangent(args)

    assert result["complete"]
    assert result["protocol"]["groups"] == 1
    assert len(result["models"]["mse_s0"]["groups"]) == 1
    assert len(result["models"]["dpsae_s0"]["groups"]) == 1
    assert result["inputs"]["natural_cache"]["sha256"] == mechanism.sha256(cache)
    assert result["paired"]["mse_s0:dpsae_s0"]["diagnostics"][
        "exact_favors_dpsae"
    ]
    assert output.exists()


def _model_payload(method: str) -> dict:
    model = BatchTopKSAE(3, 4, 1, seed=5)
    with torch.no_grad():
        model.activation_threshold.fill_(0.1)
        model.threshold_updates.fill_(1)
    return {
        "spec": {"method": method, "seed": 5, "k": 1},
        "state_dict": model.state_dict(),
    }


def test_nonorth_stage_runs_from_frozen_model_and_activation_caches(tmp_path):
    generator = torch.Generator().manual_seed(37)
    activations = torch.randn(2, 4, 3, generator=generator)
    cache = tmp_path / "natural.pt"
    models = tmp_path / "models.pt"
    output = tmp_path / "nonorth.json"
    torch.save({"activations": activations.half()}, cache)
    torch.save(
        {
            "mse_s5": _model_payload("mse"),
            "dpsae_s5": _model_payload("dpsae"),
        },
        models,
    )
    args = Namespace(
        natural_cache=cache,
        static_calibration=None,
        ridge=0.2,
        group_size=4,
        exact_tokens=8,
        max_groups=2,
        bootstrap_samples=16,
        seed=6,
        pair=None,
        models=models,
        device="cpu",
        output=output,
    )

    result = mechanism.run_nonorth(args)

    assert result["complete"]
    assert result["protocol"]["streaming"] == "one model and one geometry group at a time"
    assert result["inputs"]["models"]["sha256"] == mechanism.sha256(models)
    for model in result["models"].values():
        assert len(model["groups"]) == 2
        assert all(
            row["orthogonal_gram_min_eigenvalue"] >= -1e-5
            for row in model["groups"]
        )
    assert output.exists()


def test_prepare_stage_caches_one_exact_reconstruction_per_paired_model(tmp_path):
    activations = torch.randn(2, 4, 3)
    cache = tmp_path / "natural.pt"
    models = tmp_path / "models.pt"
    reconstruction_dir = tmp_path / "reconstructions"
    output = tmp_path / "prepare.json"
    torch.save({"activations": activations.half()}, cache)
    torch.save(
        {
            "mse_s5": _model_payload("mse"),
            "dpsae_s5": _model_payload("dpsae"),
        },
        models,
    )
    result = mechanism.run_prepare(
        Namespace(
            natural_cache=cache,
            group_size=4,
            exact_tokens=8,
            max_groups=2,
            models=models,
            pair=None,
            device="cpu",
            batch_tokens=3,
            reconstruction_dir=reconstruction_dir,
            output=output,
        )
    )

    assert result["complete"]
    assert result["summary"] == {"model_count": 2, "pairs": 1}
    for name in ("mse_s5", "dpsae_s5"):
        reconstruction = torch.load(
            reconstruction_dir / f"{name}.pt", weights_only=False
        )
        assert reconstruction.shape == (8, 3)


def test_cli_exposes_only_auditable_stages():
    parser = mechanism.build_parser()

    assert parser.parse_args(["prepare", "--ridge", "0.2"]).stage == "prepare"
    assert parser.parse_args(["tangent", "--ridge", "0.2"]).stage == "tangent"
    assert parser.parse_args(["nonorth", "--ridge", "0.2"]).stage == "nonorth"
    with pytest.raises(SystemExit):
        parser.parse_args(["support"])
