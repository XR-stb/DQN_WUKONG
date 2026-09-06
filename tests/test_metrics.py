from __future__ import annotations

import multiprocessing as mp

from wukong_rl.metrics import JsonlMetricWriter, emit_metric, metric_worker


def test_metric_process_writes_separate_streams(tmp_path) -> None:
    context = mp.get_context("spawn")
    metric_queue = context.Queue(maxsize=8)
    process = context.Process(target=metric_worker, args=(tmp_path, metric_queue))
    process.start()
    assert emit_metric(metric_queue, "actor", "step", environment_fps=8.0)
    assert emit_metric(metric_queue, "train", "learner", td_loss=1.0)
    metric_queue.put(None)
    process.join(timeout=10)
    assert process.exitcode == 0
    assert (tmp_path / "actor.jsonl").exists()
    assert (tmp_path / "train.jsonl").exists()


def test_locked_latest_snapshot_does_not_stop_jsonl_stream(tmp_path, monkeypatch) -> None:
    writer = JsonlMetricWriter(tmp_path, "actor")

    def locked(*_args, **_kwargs):
        raise PermissionError("locked by reader")

    monkeypatch.setattr("wukong_rl.metrics.os.replace", locked)
    monkeypatch.setattr("wukong_rl.metrics.time.sleep", lambda _seconds: None)

    writer.write("step", environment_steps=1)
    writer.write("step", environment_steps=2)

    lines = (tmp_path / "actor.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert writer.latest_snapshot_failures == 2
