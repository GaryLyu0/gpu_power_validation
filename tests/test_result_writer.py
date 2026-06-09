from runner.case_schema import parse_case
from runner.result_writer import compute_sweep_power_stats, parse_workload_results
from runner.telemetry import TelemetrySample


def test_parse_workload_json_lines(tmp_path) -> None:
    stdout = tmp_path / "stdout.log"
    stdout.write_text(
        "not json\n"
        '{"mode":"h2d","phase":"presweep","bytes":4096,"bandwidth_gbps":1.0}\n'
        '{"mode":"h2d","phase":"steady","bytes":268435456,"bandwidth_gbps":900.0}\n',
        encoding="utf-8",
    )

    results = parse_workload_results(stdout)

    assert len(results) == 2
    assert results[-1]["phase"] == "steady"
    assert results[-1]["bandwidth_gbps"] == 900.0


def test_compute_sweep_power_stats_uses_steady_windows() -> None:
    case = parse_case(
        {
            "case_id": "unit_sweep",
            "title": "unit sweep",
            "category": "core",
            "platform": {"gpus": [0]},
            "workload": {"type": "command", "executable": "echo"},
            "parameters": {"warmup_sec": 2, "steady_sec": 8},
            "telemetry": {"sampler": "mock_nvml", "interval_ms": 1000, "metrics": []},
        }
    )
    workload_results = [
        {
            "phase": "runner_sweep_point_start",
            "label": "000_duty_cycle-0pct",
            "point": {"duty_cycle": "0.0", "duty_cycle_label": "0pct"},
            "timestamp_ns": 0,
        },
        {
            "phase": "runner_sweep_point_end",
            "label": "000_duty_cycle-0pct",
            "point": {"duty_cycle": "0.0", "duty_cycle_label": "0pct"},
            "timestamp_ns": 10_000_000_000,
            "return_code": 0,
        },
        {
            "phase": "runner_sweep_point_start",
            "label": "001_duty_cycle-100pct",
            "point": {"duty_cycle": "1.0", "duty_cycle_label": "100pct"},
            "timestamp_ns": 10_000_000_000,
        },
        {
            "phase": "runner_sweep_point_end",
            "label": "001_duty_cycle-100pct",
            "point": {"duty_cycle": "1.0", "duty_cycle_label": "100pct"},
            "timestamp_ns": 20_000_000_000,
            "return_code": 0,
        },
    ]
    samples = [
        _sample(1_000_000_000, 100.0),
        _sample(3_000_000_000, 120.0),
        _sample(4_000_000_000, 140.0),
        _sample(11_000_000_000, 200.0),
        _sample(13_000_000_000, 220.0),
        _sample(19_000_000_000, 240.0),
    ]

    sweep_stats = compute_sweep_power_stats(case, samples, workload_results)

    assert len(sweep_stats) == 2
    assert sweep_stats[0]["label"] == "000_duty_cycle-0pct"
    assert sweep_stats[0]["point"]["duty_cycle"] == "0.0"
    assert sweep_stats[0]["steady_start_timestamp_ns"] == 2_000_000_000
    assert sweep_stats[0]["sample_count"] == 2
    assert sweep_stats[0]["power"]["avg_w"] == 130.0
    assert sweep_stats[0]["power"]["p95_w"] == 139.0
    assert sweep_stats[0]["power"]["max_w"] == 140.0
    assert sweep_stats[0]["steady_window_fallback_used"] is False

    assert sweep_stats[1]["label"] == "001_duty_cycle-100pct"
    assert sweep_stats[1]["point"]["duty_cycle"] == "1.0"
    assert sweep_stats[1]["steady_start_timestamp_ns"] == 12_000_000_000
    assert sweep_stats[1]["sample_count"] == 2
    assert sweep_stats[1]["power"]["avg_w"] == 230.0
    assert sweep_stats[1]["power"]["p95_w"] == 239.0
    assert sweep_stats[1]["power"]["max_w"] == 240.0
    assert sweep_stats[1]["steady_window_fallback_used"] is False


def _sample(timestamp_ns: int, power_w: float) -> TelemetrySample:
    return TelemetrySample(
        timestamp_ns=timestamp_ns,
        gpu_index=0,
        power_w=power_w,
        temperature_c=35.0,
        graphics_clock_mhz=1410,
        memory_clock_mhz=1593,
        utilization_gpu_percent=50,
        utilization_memory_percent=10,
        memory_used_mb=512.0,
        memory_total_mb=81920.0,
    )
