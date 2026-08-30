from __future__ import annotations

import pytest

from wukong_rl.config import CaptureConfig, PipelineConfig


def test_dxcam_requires_explicit_osd_safety_acknowledgement() -> None:
    config = PipelineConfig(capture=CaptureConfig(backend="dxcam"))
    with pytest.raises(ValueError, match="dxcam_osd_disabled"):
        config.validate()
