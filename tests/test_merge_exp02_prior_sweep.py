import csv
import json

from scripts.merge_exp02_prior_sweep import TABLES, merge_seed_directories


def test_merge_prior_sweep_preserves_seed_provenance(tmp_path) -> None:
    source = tmp_path / "source"
    for seed in (0, 1):
        directory = source / f"seed{seed}"
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "config": {"seeds": [seed], "shared": 3},
                    "git_revision": "abc123",
                    "crossover_weight": 2.5,
                    "elapsed_seconds": 10 + seed,
                }
            )
        )
        for table in TABLES:
            with (directory / f"{table}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["seed", "value"])
                writer.writeheader()
                writer.writerow({"seed": seed, "value": f"{table}-{seed}"})

    output = tmp_path / "merged"
    metadata = merge_seed_directories(source, output, [0, 1])

    assert metadata["complete"]
    assert metadata["config"]["seeds"] == [0, 1]
    assert metadata["git_revision"] == "abc123"
    assert metadata["sum_seed_elapsed_seconds"] == 21
    for table in TABLES:
        with (output / f"{table}.csv").open(newline="") as handle:
            assert [int(row["seed"]) for row in csv.DictReader(handle)] == [0, 1]
