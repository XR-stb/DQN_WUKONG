# WukongTelemetry

This is a read-only B1CSharpLoader mod. It samples player/lock-target state on
the Unreal game thread and publishes the newest JSON snapshot from a separate
thread through `\\.\pipe\wukong_rl_telemetry`.

The mod does not change attributes, invoke actions, or write to the save game.
`skill_ids.txt` contains four comma-separated skill IDs; zero disables exact
readiness for that slot until its ID has been calibrated.

Continuous capture uses one persistent Unreal `FTicker`, which requires the
loader's JIT mode. Installation therefore requires the explicit `-EnableJit`
switch; the installer backs up the existing loader configuration first. In AOT
mode the mod fails closed without opening its pipe.

Build from the repository root with:

```powershell
dotnet build .\mods\WukongTelemetry\WukongTelemetry.csproj -c Release
```

By default the project uses the pinned `game-a11y/B1CSharpLoader` checkout
prepared by the setup script. `LoaderSourceDir` can be overridden when it is
stored elsewhere. Loader/game installation is deliberately separate from the
build so compiling cannot mutate a live game directory.
