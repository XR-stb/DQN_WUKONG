from pathlib import Path


def test_telemetry_mod_reuses_loader_game_thread_queue() -> None:
    mod_directory = Path(__file__).parents[1] / "mods" / "WukongTelemetry"
    source = (mod_directory / "TelemetryMod.cs").read_text(encoding="utf-8")
    dispatcher = (mod_directory / "GameThreadDispatcher.cs").read_text(encoding="utf-8")

    assert "new GameThreadDispatcher(CaptureOnGameThread)" in source
    assert '"UnrealEngine.GameThreadHelper"' in dispatcher
    assert "Delegate.CreateDelegate(" in dispatcher
    assert "FThreading.RunOnGameThread(" not in source
    assert "FThreading.RunOnGameThread(" not in dispatcher
    assert "FTicker.AddTicker(" not in source
