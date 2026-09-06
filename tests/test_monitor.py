from __future__ import annotations

import json
import time

import numpy as np

from wukong_rl.monitor import (
    _latest_counter_session,
    _recent_jsonl,
    render_snapshot,
    run_monitor,
)


def test_recent_jsonl_ignores_partial_tail_and_limits_history(tmp_path) -> None:
    path = tmp_path / "actor.jsonl"
    with path.open("wb") as stream:
        for index in range(5):
            stream.write((json.dumps({"step": index}) + "\n").encode())
        stream.write(b'{"partial":')

    events = _recent_jsonl(path, 3)

    assert [event["step"] for event in events] == [3, 4]


def test_latest_counter_session_removes_previous_training_run() -> None:
    events = [
        {"environment_steps": 4600},
        {"environment_steps": 4601},
        {"environment_steps": 24177},
        {"environment_steps": 24178},
    ]

    current = _latest_counter_session(events, "environment_steps")

    assert [event["environment_steps"] for event in current] == [24177, 24178]


def test_monitor_ctrl_c_exits_without_propagating_traceback(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "wukong_rl.monitor.time.sleep",
        lambda _seconds: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    run_monitor(tmp_path, tmp_path, refresh_seconds=1.0)

    assert "训练进程不受影响" in capsys.readouterr().out


def test_render_snapshot_surfaces_trend_collapse_and_stale_metrics() -> None:
    snapshot = {
        "timestamp": time.time(),
        "actor_age": 60.0,
        "learner_age": 60.0,
        "actor": {},
        "learner": {},
        "learner_summary": {},
        "latest_episode_metric": {},
        "replay": {
            "available": True,
            "next_global_id": 200,
            "current": {
                "episode_id": 2,
                "steps": 20,
                "reward": -2.0,
                "boss_health": 80.0,
                "self_health": 50.0,
            },
            "movement": [("BACK_LEFT", 0.4)],
            "combat": [("NONE", 0.5)],
            "boss_invalid_rate": 0.1,
        },
        "recent": {
            "count": 10.0,
            "wins": 0.0,
            "damage_mean": 20.0,
            "damage_median": 19.0,
            "damage_best": 40.0,
            "reward_mean": -20.0,
            "steps_mean": 500.0,
        },
        "previous": {
            "count": 10.0,
            "wins": 0.0,
            "damage_mean": 21.0,
            "damage_median": 20.0,
            "damage_best": 41.0,
            "reward_mean": -19.0,
            "steps_mean": 450.0,
        },
    }

    rendered = render_snapshot(snapshot)

    assert "STOPPED" in rendered
    assert "尚未形成一致的上升趋势" in rendered
    assert "移动策略偏向 BACK_LEFT" in rendered
    assert "Boss 血量低置信度比例" in rendered
