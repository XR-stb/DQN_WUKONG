from pathlib import Path


def test_telemetry_mod_patches_existing_loader_tick_once() -> None:
    mod_directory = Path(__file__).parents[1] / "mods" / "WukongTelemetry"
    source = (mod_directory / "TelemetryMod.cs").read_text(encoding="utf-8")

    assert "if (!LoaderJitEnabled())" in source
    assert 'new Harmony("wukong_rl.telemetry.tick")' in source
    assert "_harmony.Patch(tickMethod" in source
    assert "GameThreadTickPostfix" in source
    assert "CapturePeriodSeconds" in source
    assert "FThreading.RunOnGameThread(" not in source
    assert "FTicker.AddTicker(" not in source
