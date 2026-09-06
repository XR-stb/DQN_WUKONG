from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from wukong_rl.config import CaptureConfig, EnvironmentConfig, PipelineConfig, TelemetryConfig


def test_dxcam_requires_explicit_osd_safety_acknowledgement() -> None:
    config = PipelineConfig(capture=CaptureConfig(backend="dxcam"))
    with pytest.raises(ValueError, match="dxcam_osd_disabled"):
        config.validate()


def test_telemetry_config_rejects_unsafe_pipe_names_and_wrong_skill_count() -> None:
    config = PipelineConfig(telemetry=TelemetryConfig(pipe_name=r"folder\\pipe"))
    with pytest.raises(ValueError, match="plain Windows pipe name"):
        config.validate()

    config = PipelineConfig(telemetry=TelemetryConfig(skill_ids=[1, 2]))
    with pytest.raises(ValueError, match="four non-negative"):
        config.validate()


def test_restart_orchestration_does_not_invalidate_policy_checkpoint() -> None:
    baseline = PipelineConfig().fingerprint()
    changed = PipelineConfig(
        environment=EnvironmentConfig(
            restart_recovery_action="ANOTHER_RECOVERY",
            restart_retry_attempts=7,
            restart_retry_timeout_seconds=12.0,
        )
    )

    assert changed.fingerprint() == baseline


def test_rematch_macro_normalizes_selection_and_confirms_twice() -> None:
    path = Path(__file__).parents[1] / "config" / "actions_conf.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    sequence = payload["actions"]["FUZHAN_STAND_RESTART"]

    def flatten(items):
        for item in items:
            if isinstance(item, list) and item and isinstance(item[0], list):
                yield from flatten(item)
            else:
                yield item

    flattened = list(flatten(sequence))
    assert sum(item[:2] == ["press", "up"] for item in flattened) == 8
    assert sum(item[:2] == ["press", "e"] for item in flattened) == 2
