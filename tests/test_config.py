from __future__ import annotations

import pytest

from wukong_rl.config import CaptureConfig, PipelineConfig, TelemetryConfig


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
