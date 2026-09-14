from pathlib import Path


def test_jit_install_is_explicit_and_backed_up() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "install_telemetry_mod.ps1"
    ).read_text(encoding="utf-8")

    assert "[switch]$EnableJit" in script
    assert "if ($EnableJit)" in script
    assert "Copy-Item -LiteralPath $targetLoaderConfig -Destination $backupDirectory" in script
    assert "'EnableJit=1'" in script


def test_prebuilt_install_still_verifies_the_loader_and_does_not_require_build() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "install_telemetry_mod.ps1"
    ).read_text(encoding="utf-8")

    assert "[switch]$UsePrebuilt" in script
    assert "if ($UsePrebuilt)" in script
    assert "$loaderSha256" in script
    assert "Prebuilt telemetry package is missing" in script


def test_collaboration_export_excludes_runtime_state() -> None:
    script = (
        Path(__file__).parents[1] / "scripts" / "export_collaboration_bundle.ps1"
    ).read_text(encoding="utf-8")

    assert "artifacts\\datasets-telemetry-clean\\yinhu" in script
    assert "artifacts\\checkpoints\\$name" in script
    assert "artifacts/replay" in script
    assert "COLLABORATION-MANIFEST.json" in script
