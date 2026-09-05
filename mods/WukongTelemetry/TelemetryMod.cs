using System;
using System.Globalization;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using b1;
using BtlShare;
using CSharpModBase;
using UnrealEngine;
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
        private const int SchemaVersion = 1;
        private const float CapturePeriodSeconds = 0.1f;
        private const long UnixEpochTicks = 621355968000000000L;

        private readonly object _snapshotLock = new object();
        private readonly int[] _skillIds = new int[4];
        private FTickerDelegate? _captureTicker;
        private Thread? _pipeThread;
        private NamedPipeServerStream? _pipe;
        private volatile bool _stopping;
        private long _sequence;
        private long _publishedSequence;
        private string? _latestJson;
        private int? _lastSkillId;
        private BUS_GSEventCollection? _events;
        private int? _subscribedPlayerId;
        private DateTime _skillEventRetryAfterUtc = DateTime.MinValue;

        public string Name => "WukongTelemetry";
        public string Version => "0.1.0";

        public void Init()
        {
            _stopping = false;
            if (SharedRuntimeState.IsAOT)
            {
                Console.WriteLine(
                    $"[{Name}] disabled: telemetry ticker requires CSharpLoader EnableJit=1");
                return;
            }
            LoadSkillIds();
            _pipeThread = new Thread(PipeLoop)
            {
                IsBackground = true,
                Name = "WukongTelemetryPipe"
            };
            _pipeThread.Start();
            // Register one persistent native-to-managed callback. Repeated
            // FThreading.RunOnGameThread calls exhaust Mono's trampoline pool.
            _captureTicker = CaptureTick;
            FTicker.AddTicker(_captureTicker, CapturePeriodSeconds);
            Console.WriteLine($"[{Name}] read-only telemetry started on \\\\.\\pipe\\{PipeName}");
        }

        public void DeInit()
        {
            _stopping = true;
            if (_captureTicker != null)
            {
                try { FTicker.RemoveTicker(_captureTicker); }
                catch { }
                _captureTicker = null;
            }
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
                builder.Append("{\"slot\":").Append(slot)
                    .Append(",\"skill_id\":").Append(skillId)
                    .Append(",\"ready\":");
                if (player == null || skillId <= 0) builder.Append("null");
                else AppendBool(builder, BGUFunctionLibraryCS.BGUIsSkillReady(player, skillId));
                builder.Append(",\"active\":");
                if (player == null || skillId <= 0) builder.Append("null");
                else AppendBool(builder, BGUFunctionLibraryCS.BGUIsSkillActive(player, skillId));
                builder.Append('}');
            }
            builder.Append("],\"last_skill_id\":");
            if (_lastSkillId.HasValue) builder.Append(_lastSkillId.Value);
            else builder.Append("null");
            return builder.Append('}').ToString();
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
            }
            builder.Append('}');
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
