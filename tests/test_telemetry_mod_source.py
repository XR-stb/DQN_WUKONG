from pathlib import Path


def test_telemetry_mod_uses_one_persistent_game_thread_ticker() -> None:
    source = (
        Path(__file__).parents[1] / "mods" / "WukongTelemetry" / "TelemetryMod.cs"
    ).read_text(encoding="utf-8")

    assert "FTicker.AddTicker(_captureTicker" in source
    assert "FTicker.RemoveTicker(_captureTicker" in source
    assert "FThreading.RunOnGameThread(" not in source
    assert "new Timer(" not in source
