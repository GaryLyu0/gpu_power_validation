from runner.result_writer import parse_workload_results


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
