import csv

import matplotlib.pyplot as plt
import pytest

from dpsae import closure_plots as plots
from dpsae.plot_style import FIGURE_FONT_FAMILY, paper_context


def _concept_block():
    records = []
    for method, offset in (("mse", 0.0), ("dpsae", 0.02)):
        for index, stage in enumerate(plots.STAGES):
            estimate = 0.8 - 0.03 * index + offset
            records.append(
                {
                    "method": method,
                    "stage": stage,
                    "estimate": estimate,
                    "ci_low": estimate - 0.01,
                    "ci_high": estimate + 0.01,
                }
            )
    return {"available": True, "records": records}


def test_concept_scaffold_renders_exact_size_pdf_and_png(tmp_path):
    pdf, png = plots.plot_concept_ladder(_concept_block(), tmp_path / "concept")
    assert pdf.is_file() and png.is_file()
    image = plt.imread(png)
    assert image.shape[1] == 1650
    assert image.shape[0] == 726
    with paper_context():
        assert plt.rcParams["font.sans-serif"][0] == FIGURE_FONT_FAMILY


def test_plot_blocks_fail_closed_on_incomplete_or_unavailable_results(tmp_path):
    block = _concept_block()
    block["records"].pop()
    with pytest.raises(ValueError, match="complete method-stage grid"):
        plots.plot_concept_ladder(block, tmp_path / "concept")
    with pytest.raises(ValueError, match="unavailable"):
        plots.plot_frozen_network_noninferiority(
            {"available": False}, tmp_path / "frozen"
        )


def test_noninferiority_and_static_control_scaffolds_render(tmp_path):
    frozen = {
        "available": True,
        "noninferiority_margin": 1.01,
        "records": [
            {"seed": seed, "estimate": 0.9, "ci_low": 0.85, "ci_high": 0.95}
            for seed in (0, 1, 2)
        ],
    }
    static = {
        "available": True,
        "target_nmse_low": 0.99,
        "target_nmse_high": 1.01,
        "records": [
            {
                "candidate": "MSE",
                "method": "mse",
                "nmse_ratio": 1.0,
                "decoder_reduction": 0.0,
                "selected": False,
            },
            {
                "candidate": "DPSAE",
                "method": "dpsae",
                "nmse_ratio": 1.0,
                "decoder_reduction": 0.2,
                "selected": True,
            },
            {
                "candidate": "spectral-2",
                "method": "spectral",
                "nmse_ratio": 1.05,
                "decoder_reduction": 0.1,
                "selected": False,
            },
        ],
    }
    assert all(path.is_file() for path in plots.plot_frozen_network_noninferiority(frozen, tmp_path / "frozen"))
    assert all(path.is_file() for path in plots.plot_static_nmse_control(static, tmp_path / "static"))


def test_summary_table_has_frozen_machine_readable_columns(tmp_path):
    row = {field: field for field in plots.SUMMARY_FIELDS}
    output = plots.write_summary_table(
        {"available": True, "rows": [row]}, tmp_path / "summary.csv"
    )
    with output.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames) == plots.SUMMARY_FIELDS
        assert list(reader) == [row]


def test_payload_requires_release_binding_and_explicit_table_gate():
    payload = {
        "schema_version": 1,
        "complete": True,
        "release_manifest_sha256": "a" * 64,
        "figures": {},
        "summary_table": {"available": False},
    }
    plots.validate_payload(payload)
    payload["release_manifest_sha256"] = "short"
    with pytest.raises(ValueError, match="bound"):
        plots.validate_payload(payload)
