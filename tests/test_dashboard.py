from __future__ import annotations

import pytest

from wukong_rl.dashboard import load_events


def test_dashboard_skips_partial_jsonl_records(tmp_path) -> None:
    path = tmp_path / "actor.jsonl"
    path.write_text('{"environment_steps":1,"environment_fps":8.0}\n{"partial"', encoding="utf-8")
    assert load_events(path) == [{"environment_steps": 1, "environment_fps": 8.0}]


def test_dashboard_can_render_a_bounded_png_without_live_training(tmp_path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from wukong_rl.dashboard import run_dashboard

    output = tmp_path / "dashboard.png"
    run_dashboard(
        tmp_path / "metrics",
        replay_directory=tmp_path / "replay",
        snapshot_path=output,
    )

    assert output.exists()
    assert output.stat().st_size > 10_000
