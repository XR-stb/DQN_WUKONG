from pathlib import Path


def test_telemetry_mod_registers_one_persistent_native_ticker() -> None:
    mod_directory = Path(__file__).parents[1] / "mods" / "WukongTelemetry"
    source = (mod_directory / "TelemetryMod.cs").read_text(encoding="utf-8")

    assert "if (!LoaderJitEnabled())" in source
    assert '"UnrealEngine.Runtime.Native.Native_FTicker"' in source
    assert '"Reg_CoreTicker"' in source
    assert "RegisterPersistentTickerOnGameThread" in source
    assert "UnregisterPersistentTickerOnGameThread" in source
    assert source.count("FThreading.RunOnGameThread(") == 2
    assert "FTicker.AddTicker(" not in source
