from pathlib import Path


def test_telemetry_mod_uses_one_persistent_jit_ticker() -> None:
    mod_directory = Path(__file__).parents[1] / "mods" / "WukongTelemetry"
    source = (mod_directory / "TelemetryMod.cs").read_text(encoding="utf-8")

    assert "if (SharedRuntimeState.IsAOT)" in source
    assert "FTicker.AddTicker(_captureTicker" in source
    assert "FTicker.RemoveTicker(_captureTicker" in source
    assert "FThreading.RunOnGameThread(" not in source
