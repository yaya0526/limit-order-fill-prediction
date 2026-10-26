import json

from scripts.demo_results import (
    DEFAULT_METRICS_PATH,
    build_takeaways,
    compact_results,
    load_results,
    main,
    render_text,
)


def test_load_results_validates_published_horizons_and_confusion_counts():
    rows = load_results()

    assert [int(row["horizon_seconds"]) for row in rows] == [5, 30, 60]
    for row in rows:
        assert sum(int(row[key]) for key in ("tn", "fp", "fn", "tp")) == int(
            row["eligible_rows"]
        )


def test_text_demo_surfaces_result_and_scope_caveat():
    rows = load_results()
    rendered = render_text(rows, DEFAULT_METRICS_PATH)
    takeaways = build_takeaways(rows)

    assert "no raw data required" in rendered
    assert "60s" in rendered
    assert "synthetic teaching data" in rendered
    assert any("weakly selective" in takeaway for takeaway in takeaways)


def test_json_demo_is_machine_readable(capsys):
    assert main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload == compact_results(load_results())
    assert [row["horizon_seconds"] for row in payload] == [5, 30, 60]
