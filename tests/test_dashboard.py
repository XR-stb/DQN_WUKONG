from __future__ import annotations

from wukong_rl.dashboard import load_events


def test_dashboard_skips_partial_jsonl_records(tmp_path) -> None:
    path = tmp_path / "actor.jsonl"
    path.write_text('{"environment_steps":1,"environment_fps":8.0}\n{"partial"', encoding="utf-8")
    assert load_events(path) == [{"environment_steps": 1, "environment_fps": 8.0}]
