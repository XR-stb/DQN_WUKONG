from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wukong_rl.config import CaptureConfig, EnvironmentConfig, ModelConfig, PipelineConfig
from wukong_rl.evaluation import _build_summary, evaluate_live
from wukong_rl.types import ACTION_MASK_SIZE, ActionCommand, EpisodeResult, EpisodeState, Transition

from conftest import make_observation


def test_partial_evaluation_summary_is_not_a_pass() -> None:
    result = EpisodeResult(1, EpisodeState.WON, 10.0, 10, 1.0, 0.0, 100.0, 100.0, 0.0)
    summary = _build_summary(
        [result],
        requested_episodes=3,
        checkpoint_path="agent.pt",
        config_hash="hash",
        error="restart failed",
    )
    assert summary["episodes"] == 1
    assert summary["requested_episodes"] == 3
    assert summary["win_rate"] == 1.0
    assert not summary["complete"]
    assert not summary["passed"]


def test_live_evaluation_reports_progress_and_saves_invalid_partial_run(
    tmp_path, monkeypatch, capsys
) -> None:
    class FakeAgent:
        def initial_state(self):
            return None

        def act(self, observation, state, exploration, rng):
            return ActionCommand(), state, np.zeros(ACTION_MASK_SIZE, dtype=np.float32)

        def action_entropy(self, q_values, action_mask):
            return 0.0

    class FakeWriter:
        def isOpened(self):
            return True

        def write(self, frame):
            pass

        def release(self):
            pass

    class FakeTerminal:
        last_valid_boss = 99.0
        last_valid_self = 10.0

    class FakeEnvironment:
        def __init__(self):
            self.terminal = FakeTerminal()
            self.last_raw_frame = np.zeros((8, 8, 3), dtype=np.uint8)
            self.closed = False

        def reset(self):
            return make_observation(frame_shape=(8, 8, 3))

        def step(self, action):
            current = make_observation(frame_shape=(8, 8, 3))
            following = make_observation(
                1, state=EpisodeState.INVALID, frame_shape=(8, 8, 3)
            )
            return Transition(
                current,
                action,
                -1.0,
                following,
                False,
                True,
                1.0,
                episode_id=1,
                step_id=0,
            )

        def close(self):
            self.closed = True

    environment = FakeEnvironment()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("wukong_rl.evaluation.R2D3Agent", lambda *args, **kwargs: FakeAgent())
    monkeypatch.setattr("wukong_rl.evaluation.load_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr("wukong_rl.evaluation.build_live_environment", lambda config: environment)
    monkeypatch.setattr("wukong_rl.evaluation.cv2.VideoWriter", lambda *args: FakeWriter())

    config = PipelineConfig(
        capture=CaptureConfig(
            backend="array",
            width=8,
            height=8,
            observation_width=8,
            observation_height=8,
        ),
        environment=EnvironmentConfig(control_hz=8),
        model=ModelConfig(hidden_size=32, burn_in=1, unroll=4, n_step=1, batch_size=2),
    )
    summary = evaluate_live(config, Path("agent.pt"), episodes=2)

    output = capsys.readouterr().out
    assert "不会训练或更新权重" in output
    assert "第 1 局结束" in output
    assert "automatic restart was skipped" in output
    assert summary["episodes"] == 1
    assert summary["error"] is not None
    assert environment.closed
    summary_paths = list((tmp_path / "artifacts" / "evaluations").glob("*/summary.json"))
    assert len(summary_paths) == 1
    assert json.loads(summary_paths[0].read_text(encoding="utf-8"))["episodes"] == 1
