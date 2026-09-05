from pathlib import Path


def test_jit_install_is_explicit_and_backed_up() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "install_telemetry_mod.ps1"
    ).read_text(encoding="utf-8")

    assert "[switch]$EnableJit" in script
    assert "if ($EnableJit)" in script
    assert "Copy-Item -LiteralPath $targetLoaderConfig -Destination $backupDirectory" in script
    assert "'EnableJit=1'" in script
