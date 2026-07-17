from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/run_steps1_4_autonomous_runpod.sh"


def test_autonomous_supervisor_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(SUPERVISOR)], check=True)


def test_autonomous_supervisor_preserves_fail_closed_contracts() -> None:
    source = SUPERVISOR.read_text(encoding="utf-8")

    launch_marker = source.index("FLEET_LAUNCH_STARTED=1")
    launcher = source.index("bash scripts/run_exp10_concept_4xa40.sh")
    assert launch_marker < launcher
    assert 'if [[ "$FLEET_LAUNCH_STARTED" == "1" ]]' in source
    assert "stop_exp10_fleet" in source

    assert "WORKER_SESSIONS=(exp10-gpu0 exp10-gpu1 exp10-gpu2 exp10-gpu3)" in source
    assert 'for worker_index in "${!WORKER_SESSIONS[@]}"' in source
    assert 'if [[ -f "$worker_done" ]]' in source
    assert 'record="$(session_record "$worker_session")"' in source
    assert 'before $worker_done' in source

    assert "--query-gpu=index,memory.used" in source
    assert "for gpu_index in 0 1 2 3" in source
    assert 'gpu_blockers+=("GPU${gpu_index}=${gpu_memory}MiB")' in source

    assert ".schema_version == 7" in source
    assert ".config_digest == $config_digest" in source
    assert ".probe_seed == $probe_seed" in source
    assert ".task_count == $task_count" in source
    assert ".measured_worker_count == $measured_workers" in source
    assert ".topology.mode == $timing_topology" in source
    assert ".barrier.synchronized == true" in source
    assert '.projection.aggregation == "slowest_measured_worker"' in source
    assert ".companion_l2_path_optimization == $l2_optimization" in source
    assert ".companion_full_code_cold_C_jobs_per_worker == $cold_c_jobs" in source
    assert ".passed == (.projection.projected_pod_hours <= $maximum_pod_hours)" in source
