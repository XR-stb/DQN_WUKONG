using System;
using System.Globalization;
using System.IO;
using System.IO.Pipes;
using System.Reflection;
using System.Text;
using System.Threading;
using b1;
using BtlShare;
using CSharpModBase;
using UnrealEngine.Engine;
using UnrealEngine.Runtime;

namespace WukongTelemetry
{
    /// <summary>
    /// Read-only Black Myth telemetry bridge. It never mutates game state and
    /// never performs pipe I/O on the Unreal game thread.
    /// </summary>
    public sealed class TelemetryMod : ICSharpMod
    {
        private const string PipeName = "wukong_rl_telemetry";
        private const int SchemaVersion = 2;
        private const int SkillAvailabilityBlockMask =
            (int)ECanCastSkillResult.CCSR_NOT_ENOUGH_ATTR |
            (int)ECanCastSkillResult.CCSR_COOLDOWN |
            (int)ECanCastSkillResult.CCSR_PRECOOLDOWN |
            (int)ECanCastSkillResult.CCSR_DEAD |
            (int)ECanCastSkillResult.CCSR_OTHER |
            (int)ECanCastSkillResult.CCSR_NOSKILL |
            (int)ECanCastSkillResult.CCSR_NOT_ENOUGH_STAMINA |
            (int)ECanCastSkillResult.CCSR_INVALID_CASTER |
            (int)ECanCastSkillResult.CCSR_EMPTY_SKILLIST |
            (int)ECanCastSkillResult.CCSR_NULL_DATA |
            (int)ECanCastSkillResult.CCSR_IN_SILENT |
            (int)ECanCastSkillResult.CCSR_NO_TASKSTAGEFILTER |
            (int)ECanCastSkillResult.CCSR_IN_LANDPROTECT;
        private const float CapturePeriodSeconds = 0.1f;
        private const long UnixEpochTicks = 621355968000000000L;

        private readonly object _snapshotLock = new object();
        private readonly int[] _skillIds = new int[4];
        private FTicker? _ticker;
        private FTickerDelegate? _captureTicker;
        private Delegate? _nativeTickerRegistrar;
        private FDelegateHandle _tickerHandle;
        private Thread? _pipeThread;
        private NamedPipeServerStream? _pipe;
        private volatile bool _stopping;
        private long _sequence;
        private long _publishedSequence;
        private string? _latestJson;
        private int? _lastSkillId;
        private int? _lastSkillMappingId;
        private int? _lastSkillOriginalId;
        private int? _lastSkillSourceType;
        private long _lastSkillEventSequence;
        private BUS_GSEventCollection? _events;
        private int? _subscribedPlayerId;
        private DateTime _skillEventRetryAfterUtc = DateTime.MinValue;

        public string Name => "WukongTelemetry";
        public string Version => "0.2.0";

        public void Init()
        {
            _stopping = false;
            if (!LoaderJitEnabled())
            {
                Console.WriteLine(
                    $"[{Name}] disabled: telemetry ticker requires CSharpLoader EnableJit=1");
                return;
            }
            LoadSkillIds();
            try
            {
                _captureTicker = CaptureTick;
                // Registration crosses to the game thread once. Sampling is
                // then driven by one persistent native ticker callback.
                FThreading.RunOnGameThread(RegisterPersistentTickerOnGameThread);
            }
            catch (Exception error)
            {
                _ticker = null;
                _captureTicker = null;
                _nativeTickerRegistrar = null;
                Console.WriteLine($"[{Name}] disabled: persistent ticker registration failed: {error}");
                return;
            }
            _pipeThread = new Thread(PipeLoop)
            {
                IsBackground = true,
                Name = "WukongTelemetryPipe"
            };
            _pipeThread.Start();
            Console.WriteLine($"[{Name}] read-only telemetry started on \\\\.\\pipe\\{PipeName}");
        }

        public void DeInit()
        {
            _stopping = true;
            if (_ticker != null)
            {
                // One matching game-thread hop at unload; never per sample.
                try { FThreading.RunOnGameThread(UnregisterPersistentTickerOnGameThread); }
                catch { }
            }
            _ticker = null;
            _captureTicker = null;
            _nativeTickerRegistrar = null;
            UnsubscribeSkillEvent();
            try { _pipe?.Dispose(); } catch { }
            _pipe = null;
            _pipeThread?.Join(1000);
            _pipeThread = null;
            Console.WriteLine($"[{Name}] stopped");
        }

        private bool CaptureTick(float _)
        {
            if (_stopping)
                return false;
            try
            {
                CaptureOnGameThread();
            }
            catch (Exception error)
            {
                Console.WriteLine($"[{Name}] capture error: {error.Message}");
            }
            return !_stopping;
        }

        private void RegisterPersistentTickerOnGameThread()
        {
            if (_captureTicker == null)
                throw new InvalidOperationException("capture ticker delegate is unavailable");

            var tickerType = typeof(FTicker);
            _ticker = new FTicker();
            var delegateField = tickerType.GetField(
                    "del", BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingFieldException(tickerType.FullName, "del");
            delegateField.SetValue(_ticker, _captureTicker);
            var callback = tickerType.GetField(
                    "callback", BindingFlags.Instance | BindingFlags.NonPublic)
                ?.GetValue(_ticker)
                ?? throw new MissingFieldException(tickerType.FullName, "callback");

            var nativeTickerType = tickerType.Assembly.GetType(
                "UnrealEngine.Runtime.Native.Native_FTicker", throwOnError: true);
            _nativeTickerRegistrar = nativeTickerType.GetField(
                    "Reg_CoreTicker", BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic)
                ?.GetValue(null) as Delegate
                ?? throw new MissingFieldException(nativeTickerType.FullName, "Reg_CoreTicker");

            var arguments = new object[]
            {
                IntPtr.Zero,
                callback,
                _tickerHandle,
                (csbool)true,
                CapturePeriodSeconds,
            };
            _nativeTickerRegistrar.DynamicInvoke(arguments);
            _tickerHandle = (FDelegateHandle)arguments[2];
            if (_tickerHandle.ID == 0)
                throw new InvalidOperationException("native ticker returned an empty handle");
        }

        private void UnregisterPersistentTickerOnGameThread()
        {
            if (_ticker == null || _nativeTickerRegistrar == null || _tickerHandle.ID == 0)
                return;
            var callback = typeof(FTicker).GetField(
                    "callback", BindingFlags.Instance | BindingFlags.NonPublic)
                ?.GetValue(_ticker);
            if (callback == null)
                return;
            var arguments = new object[]
            {
                IntPtr.Zero,
                callback,
                _tickerHandle,
                (csbool)false,
                0f,
            };
            _nativeTickerRegistrar.DynamicInvoke(arguments);
            _tickerHandle = default;
        }

        private static bool LoaderJitEnabled()
        {
            try
            {
                var path = Path.Combine(
                    AppDomain.CurrentDomain.BaseDirectory ?? ".",
                    "CSharpLoader",
                    "b1cs.ini");
                if (!File.Exists(path))
                    return false;
                foreach (var line in File.ReadAllLines(path))
                {
                    if (string.Equals(line.Trim(), "EnableJit=1", StringComparison.OrdinalIgnoreCase))
                        return true;
                }
            }
            catch { }
            return false;
        }

        private static UWorld? GetWorld()
        {
            var reference = GCHelper.FindRef(FGlobals.GWorld);
            return reference?.Managed as UWorld;
        }

        private static APawn? GetPlayer()
        {
            var world = GetWorld();
            if (world == null)
                return null;
            return UGSE_EngineFuncLib.GetFirstLocalPlayerController(world)?.GetControlledPawn();
        }

        private void CaptureOnGameThread()
        {
            if (_stopping)
                return;
            var player = GetPlayer();
            EnsureSkillEvent(player);
            AActor? target = null;
            if (player != null)
            {
                target = BGUFunctionLibraryCS.GetUnitLockTargetActor(player);
                if (target == null)
                    target = BGUFunctionLibraryCS.BGUGetTarget(player);
            }

            var sequence = Interlocked.Increment(ref _sequence);
            var emittedUnixNs = (DateTime.UtcNow.Ticks - UnixEpochTicks) * 100L;
            var json = BuildPacket(sequence, emittedUnixNs, player, target);
            lock (_snapshotLock)
            {
                _latestJson = json;
                _publishedSequence = sequence;
            }
        }

        private void EnsureSkillEvent(APawn? player)
        {
            if (player == null)
            {
                UnsubscribeSkillEvent();
                return;
            }
            if (_events == null && DateTime.UtcNow < _skillEventRetryAfterUtc)
                return;
            var playerId = BGUFunctionLibraryCS.BGUGetUniqueID(player);
            if (_events != null && _subscribedPlayerId == playerId)
                return;
            UnsubscribeSkillEvent();
            try
            {
                _events = BUS_EventCollectionCS.Get(player);
                if (_events != null)
                {
                    _events.Evt_UnitCastSkillSuccess += OnSkillCastSuccess;
                    _subscribedPlayerId = playerId;
                    _skillEventRetryAfterUtc = DateTime.MinValue;
                }
            }
            catch (Exception error)
            {
                _events = null;
                _subscribedPlayerId = null;
                _skillEventRetryAfterUtc = DateTime.UtcNow.AddSeconds(5);
                Console.WriteLine($"[{Name}] skill event unavailable: {error.Message}");
            }
        }

        private void UnsubscribeSkillEvent()
        {
            if (_events != null)
            {
                try { _events.Evt_UnitCastSkillSuccess -= OnSkillCastSuccess; }
                catch { }
            }
            _events = null;
            _subscribedPlayerId = null;
        }

        private void OnSkillCastSuccess(
            int mappingSkillId, int originalSkillId, ECastSkillSourceType sourceType)
        {
            _lastSkillId = mappingSkillId > 0 ? mappingSkillId : originalSkillId;
            _lastSkillMappingId = mappingSkillId;
            _lastSkillOriginalId = originalSkillId;
            _lastSkillSourceType = (int)sourceType;
            _lastSkillEventSequence++;
        }

        private string BuildPacket(long sequence, long emittedUnixNs, APawn? player, AActor? target)
        {
            var builder = new StringBuilder(1024);
            builder.Append("{\"schema_version\":").Append(SchemaVersion)
                .Append(",\"sequence\":").Append(sequence)
                .Append(",\"emitted_unix_ns\":").Append(emittedUnixNs)
                .Append(",\"player\":");
            AppendEntity(builder, player, includeResources: true);
            builder.Append(",\"target\":");
            AppendEntity(builder, target, includeResources: false);
            builder.Append(",\"skills\":[");
            for (var slot = 0; slot < _skillIds.Length; slot++)
            {
                if (slot > 0) builder.Append(',');
                var skillId = _skillIds[slot];
                AppendSkill(builder, slot, skillId, player);
            }
            builder.Append("],\"last_skill_id\":");
            if (_lastSkillId.HasValue) builder.Append(_lastSkillId.Value);
            else builder.Append("null");
            builder.Append(",\"last_skill_mapping_id\":");
            if (_lastSkillMappingId.HasValue) builder.Append(_lastSkillMappingId.Value);
            else builder.Append("null");
            builder.Append(",\"last_skill_original_id\":");
            if (_lastSkillOriginalId.HasValue) builder.Append(_lastSkillOriginalId.Value);
            else builder.Append("null");
            builder.Append(",\"last_skill_source_type\":");
            if (_lastSkillSourceType.HasValue) builder.Append(_lastSkillSourceType.Value);
            else builder.Append("null");
            builder.Append(",\"last_skill_event_sequence\":")
                .Append(_lastSkillEventSequence);
            return builder.Append('}').ToString();
        }

        private static void AppendSkill(
            StringBuilder builder, int slot, int skillId, AActor? player)
        {
            builder.Append("{\"slot\":").Append(slot)
                .Append(",\"skill_id\":").Append(skillId)
                .Append(",\"ready\":");
            if (player == null || skillId <= 0)
            {
                builder.Append("null,\"active\":null,\"in_cooldown\":null")
                    .Append(",\"castable_now\":null,\"can_cast_result\":null}");
                return;
            }

            // CheckSkillCanCast includes transient animation/movement reasons.
            // Keep those raw and derive "ready" only from persistent blockers,
            // so an 8 Hz controller may still queue a skill during a combo window.
            var inCooldown = BGU_CommonUtil.IsSkillInCoolDown(skillId, player);
            var canCastResult = BGU_CommonUtil.CheckSkillCanCast(player, skillId, skillId);
            var resultBits = (int)canCastResult;
            var ready = !inCooldown && (resultBits & SkillAvailabilityBlockMask) == 0;
            AppendBool(builder, ready);
            builder.Append(",\"active\":");
            AppendBool(builder, BGUFunctionLibraryCS.BGUIsSkillActive(player, skillId));
            builder.Append(",\"in_cooldown\":");
            AppendBool(builder, inCooldown);
            builder.Append(",\"castable_now\":");
            AppendBool(builder, canCastResult == ECanCastSkillResult.CCSR_OK);
            builder.Append(",\"can_cast_result\":").Append(resultBits).Append('}');
        }

        private static void AppendEntity(StringBuilder builder, AActor? actor, bool includeResources)
        {
            if (actor == null)
            {
                builder.Append("{\"valid\":false}");
                return;
            }
            builder.Append("{\"valid\":true")
                .Append(",\"res_id\":").Append(BGUFunctionLibraryCS.BGUGetResID(actor))
                .Append(",\"unique_id\":").Append(BGUFunctionLibraryCS.BGUGetUniqueID(actor))
                .Append(",\"hp\":");
            AppendFloat(builder, BGUFunctionLibraryCS.GetAttrValue(actor, EBGUAttrFloat.Hp));
            builder.Append(",\"hp_max\":");
            AppendFloat(builder, BGUFunctionLibraryCS.GetAttrValue(actor, EBGUAttrFloat.HpMax));
            builder.Append(",\"dead\":");
            AppendBool(builder, BGUFunctionLibraryCS.BGUIsUnitDead(actor));
            builder.Append(",\"in_battle\":");
            AppendBool(builder, BGUFunctionLibraryCS.BGUIsUnitInBattle(actor));

            if (includeResources)
            {
                AppendAttribute(builder, actor, "mp", EBGUAttrFloat.Mp);
                AppendAttribute(builder, actor, "mp_max", EBGUAttrFloat.MpMax);
                AppendAttribute(builder, actor, "stamina", EBGUAttrFloat.Stamina);
                AppendAttribute(builder, actor, "stamina_max", EBGUAttrFloat.StaminaMax);
                AppendAttribute(builder, actor, "focus_level", EBGUAttrFloat.Pelevel);
                AppendAttribute(builder, actor, "focus_level_max", EBGUAttrFloat.PelevelMax);
                AppendAttribute(builder, actor, "focus_value", EBGUAttrFloat.Pevalue);
                AppendAttribute(builder, actor, "focus_value_max", EBGUAttrFloat.PevalueMax);
                AppendAttribute(builder, actor, "fabao_energy", EBGUAttrFloat.FabaoEnergy);
                AppendAttribute(builder, actor, "fabao_energy_max", EBGUAttrFloat.FabaoEnergyMax);
                AppendAttribute(builder, actor, "vigor_energy", EBGUAttrFloat.VigorEnergy);
                AppendAttribute(builder, actor, "vigor_energy_max", EBGUAttrFloat.VigorEnergyMax);
                builder.Append(",\"can_move_run\":");
                AppendBool(builder, BGUFunctionLibraryCS.BGUCanMoveRun(actor));
                builder.Append(",\"can_move_rotate\":");
                AppendBool(builder, BGUFunctionLibraryCS.BGUCanMoveRotate(actor));
                AppendUnitState(builder, actor, "attacking", EBGUUnitState.Attacking);
                AppendUnitState(builder, actor, "attack_moving", EBGUUnitState.AttackMoving);
                AppendUnitState(builder, actor, "in_combo_window", EBGUUnitState.InComboWindow);
                AppendUnitState(builder, actor, "in_dodge_window", EBGUUnitState.InDodgeWindow);
                AppendUnitState(builder, actor, "impact_action_playing", EBGUUnitState.ImpactActionPlaying);
                AppendUnitState(builder, actor, "in_abort_window", EBGUUnitState.InAbortWindow);
                AppendSimpleState(builder, actor, "cant_attack", EBGUSimpleState.CantAttack);
                AppendSimpleState(builder, actor, "cant_move", EBGUSimpleState.CantMove);
                AppendSimpleState(builder, actor, "ignore_all_input", EBGUSimpleState.IgnoreAllInput);
            }
            builder.Append('}');
        }

        private static void AppendUnitState(
            StringBuilder builder, AActor actor, string name, EBGUUnitState state)
        {
            builder.Append(",\"").Append(name).Append("\":");
            AppendBool(builder, BGUFunctionLibraryCS.BGUHasUnitState(actor, state));
        }

        private static void AppendSimpleState(
            StringBuilder builder, AActor actor, string name, EBGUSimpleState state)
        {
            builder.Append(",\"").Append(name).Append("\":");
            AppendBool(builder, BGUFunctionLibraryCS.BGUHasUnitSimpleState(actor, state));
        }

        private static void AppendAttribute(
            StringBuilder builder, AActor actor, string name, EBGUAttrFloat attribute)
        {
            builder.Append(",\"").Append(name).Append("\":");
            AppendFloat(builder, BGUFunctionLibraryCS.GetAttrValue(actor, attribute));
        }

        private static void AppendFloat(StringBuilder builder, float value)
        {
            if (float.IsNaN(value) || float.IsInfinity(value)) builder.Append("null");
            else builder.Append(value.ToString("R", CultureInfo.InvariantCulture));
        }

        private static void AppendBool(StringBuilder builder, bool value)
        {
            builder.Append(value ? "true" : "false");
        }

        private void PipeLoop()
        {
            while (!_stopping)
            {
                try
                {
                    using (var pipe = new NamedPipeServerStream(
                        PipeName,
                        PipeDirection.Out,
                        1,
                        PipeTransmissionMode.Byte,
                        // Black Myth embeds Mono. Its Windows async named-pipe
                        // completion callback can throw repeatedly when the
                        // Python client disconnects. This method already runs
                        // on a dedicated background thread, so synchronous I/O
                        // is both sufficient and avoids touching the Mono
                        // ThreadPoolBoundHandle implementation.
                        PipeOptions.None))
                    {
                        _pipe = pipe;
                        pipe.WaitForConnection();
                        long sentSequence = -1;
                        while (!_stopping && pipe.IsConnected)
                        {
                            string? payload;
                            long currentSequence;
                            lock (_snapshotLock)
                            {
                                payload = _latestJson;
                                currentSequence = _publishedSequence;
                            }
                            if (payload != null && currentSequence != sentSequence)
                            {
                                var bytes = Encoding.UTF8.GetBytes(payload + "\n");
                                pipe.Write(bytes, 0, bytes.Length);
                                pipe.Flush();
                                sentSequence = currentSequence;
                            }
                            else
                            {
                                Thread.Sleep(10);
                            }
                        }
                    }
                }
                catch (Exception error)
                {
                    if (!_stopping)
                    {
                        Console.WriteLine($"[{Name}] pipe error: {error.Message}");
                        Thread.Sleep(250);
                    }
                }
                finally
                {
                    _pipe = null;
                }
            }
        }

        private void LoadSkillIds()
        {
            try
            {
                var path = Path.Combine(
                    AppDomain.CurrentDomain.BaseDirectory ?? ".",
                    "CSharpLoader",
                    "Mods",
                    Name,
                    "skill_ids.txt");
                if (!File.Exists(path))
                {
                    Console.WriteLine($"[{Name}] skill_ids.txt not found; skill readiness disabled");
                    return;
                }
                var parts = File.ReadAllText(path).Split(',');
                if (parts.Length != 4)
                    throw new FormatException("expected four comma-separated skill IDs");
                for (var index = 0; index < parts.Length; index++)
                {
                    if (!int.TryParse(parts[index].Trim(), out _skillIds[index]) || _skillIds[index] < 0)
                        throw new FormatException($"invalid skill ID at slot {index + 1}");
                }
            }
            catch (Exception error)
            {
                Array.Clear(_skillIds, 0, _skillIds.Length);
                Console.WriteLine($"[{Name}] invalid skill_ids.txt: {error.Message}");
            }
        }
    }
}
