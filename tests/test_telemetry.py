from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from wukong_rl.config import TelemetryConfig
from wukong_rl.telemetry import (
    HybridPerception,
    NamedPipeTelemetryClient,
    TelemetrySnapshot,
    TelemetryUnavailableError,
    snapshot_summary,
)
from wukong_rl.types import FieldMeasurement


def packet(**overrides):
    payload = {
        "schema_version": 2,
        "sequence": 7,
        "emitted_unix_ns": 123,
        "player": {
            "valid": True,
            "hp": 360.0,
            "hp_max": 400.0,
            "mp": 40.0,
            "mp_max": 100.0,
            "stamina": 75.0,
            "stamina_max": 150.0,
            "focus_level": 2.0,
            "dead": False,
            "in_battle": True,
            "can_move_run": False,
            "can_move_rotate": True,
            "attacking": True,
            "in_combo_window": False,
        },
        "target": {
            "valid": True,
            "res_id": 81102,
            "unique_id": 42,
            "hp": 25.0,
            "hp_max": 100.0,
            "dead": False,
        },
        "skills": [
            {
                "slot": 0,
                "skill_id": 101,
                "ready": True,
                "active": False,
                "in_cooldown": False,
                "castable_now": False,
                "can_cast_result": 16,
            },
            {
                "slot": 1,
                "skill_id": 102,
                "ready": False,
                "active": True,
                "in_cooldown": True,
                "castable_now": False,
                "can_cast_result": 4,
            },
        ],
        "last_skill_id": 102,
        "last_skill_mapping_id": 102,
        "last_skill_original_id": 9002,
        "last_skill_source_type": 1,
        "last_skill_event_sequence": 3,
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeScreen:
    def __init__(self):
        self.active = False
        self.closed = False

    def reset(self):
        self.active = False

    def set_episode_active(self, active):
        self.active = active

    def detect(self, _frame):
        return {
            "self_blood": FieldMeasurement(50.0, 0.8),
            "boss_blood": FieldMeasurement(80.0, 0.8),
            "self_magic": FieldMeasurement(10.0, 0.8),
            "self_energy": FieldMeasurement(20.0, 0.8),
            "skill_1": FieldMeasurement(0.0, 0.8),
            "skill_2": FieldMeasurement(1.0, 0.8),
        }


class FakeClient:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def latest(self):
        return self.snapshot

    def close(self):
        self.closed = True


def test_snapshot_schema_is_strict_and_rejects_nonfinite_values() -> None:
    snapshot = TelemetrySnapshot.from_json(packet(), received_monotonic=5.0)
    assert snapshot.player.hp == 360.0
    assert snapshot.target.res_id == 81102
    assert snapshot.skills[1].active is True
    assert snapshot.skills[1].in_cooldown is True
    assert snapshot.player.attacking is True
    assert snapshot.player.can_move_run is False
    assert snapshot.last_skill_mapping_id == 102
    assert snapshot.last_skill_original_id == 9002
    assert snapshot.last_skill_event_sequence == 3
    assert snapshot_summary(snapshot)["last_skill_original_id"] == 9002
    assert snapshot_summary(snapshot)["skills"][0]["can_cast_reasons"] == ["state"]
    assert snapshot_summary(snapshot)["skills"][1]["can_cast_reasons"] == ["cooldown"]

    with pytest.raises(ValueError, match="unsupported telemetry schema"):
        TelemetrySnapshot.from_json(packet(schema_version=3))
    with pytest.raises(ValueError, match="finite"):
        TelemetrySnapshot.from_json(packet(player={"valid": True, "hp": float("inf")}))


def test_named_pipe_client_expires_old_snapshots_without_blocking_reader() -> None:
    now = [10.0]
    config = TelemetryConfig(max_age_seconds=0.35)
    client = NamedPipeTelemetryClient(config, clock=lambda: now[0])
    client.ingest_line(packet())
    assert client.latest() is not None
    now[0] += 0.36
    assert client.latest() is None
    status = client.status()
    assert not status.fresh
    assert status.valid_packets == 1


def test_schema_one_remains_parseable_but_does_not_claim_exact_cooldown() -> None:
    legacy = json.loads(packet())
    legacy["schema_version"] = 1
    legacy["skills"] = [
        {"slot": 0, "skill_id": 101, "ready": True, "active": False}
    ]
    snapshot = TelemetrySnapshot.from_json(json.dumps(legacy))
    assert snapshot.skills[0].ready is True
    assert snapshot.skills[0].in_cooldown is None

    hybrid = HybridPerception(
        FakeScreen(),
        FakeClient(snapshot),
        TelemetryConfig(skill_ids=[101, 0, 0, 0]),
    )
    result = hybrid.detect(np.zeros((2, 2, 3), dtype=np.uint8))
    assert result["skill_1"].value == 0.0
    assert hybrid.last_sources["skill_1"] == "screen"


def test_named_pipe_client_close_never_closes_blocking_handle_cross_thread() -> None:
    entered_read = threading.Event()
    release_read = threading.Event()

    class BlockingPipe:
        closed = False

        def readline(self):
            entered_read.set()
            release_read.wait(2.0)
            return b""

        def close(self):
            self.closed = True

    pipe = BlockingPipe()
    client = NamedPipeTelemetryClient(
        TelemetryConfig(reconnect_seconds=0.05), pipe_opener=lambda _path: pipe
    )
    client.start()
    assert entered_read.wait(1.0)
    started = time.perf_counter()
    client.close()
    assert time.perf_counter() - started < 0.5
    assert not pipe.closed
    release_read.set()
    client.close()
    assert pipe.closed


def test_hybrid_perception_overrides_only_valid_configured_telemetry() -> None:
    snapshot = TelemetrySnapshot.from_json(packet(), received_monotonic=1.0)
    client = FakeClient(snapshot)
    hybrid = HybridPerception(
        FakeScreen(),
        client,
        TelemetryConfig(skill_ids=[101, 102, 0, 0], accepted_boss_res_ids=[81102]),
    )
    result = hybrid.detect(np.zeros((2, 2, 3), dtype=np.uint8))
    assert client.started
    assert result["self_blood"].value == pytest.approx(90.0)
    assert result["self_magic"].value == pytest.approx(40.0)
    assert result["self_energy"].value == pytest.approx(50.0)
    assert result["boss_blood"].value == pytest.approx(25.0)
    assert result["skill_1"].value == 1.0
    assert result["skill_2"].value == 0.0
    assert result["gunshi1"].value == 1.0
    assert result["gunshi2"].value == 1.0
    assert result["gunshi3"].value == 0.0
    assert hybrid.last_sources["boss_blood"] == "telemetry"
    hybrid.close()
    assert client.closed


def test_hybrid_perception_exposes_skill_four_active_as_control_only_state() -> None:
    raw = json.loads(packet())
    raw["skills"].append(
        {
            "slot": 3,
            "skill_id": 10424,
            "ready": True,
            "active": True,
            "in_cooldown": False,
            "castable_now": True,
            "can_cast_result": 0,
        }
    )
    snapshot = TelemetrySnapshot.from_json(json.dumps(raw), received_monotonic=1.0)
    hybrid = HybridPerception(
        FakeScreen(),
        FakeClient(snapshot),
        TelemetryConfig(skill_ids=[0, 0, 0, 10424]),
    )

    result = hybrid.detect(np.zeros((2, 2, 3), dtype=np.uint8))

    assert result["skill_4"].value == 1.0
    assert result["transformation_active"].value == 1.0
    assert hybrid.last_sources["transformation_active"] == "telemetry"


def test_hybrid_perception_falls_back_or_fails_closed_by_mode() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    fallback = HybridPerception(FakeScreen(), FakeClient(None), TelemetryConfig(mode="prefer"))
    assert fallback.detect(frame)["boss_blood"].value == 80.0

    required = HybridPerception(FakeScreen(), FakeClient(None), TelemetryConfig(mode="required"))
    with pytest.raises(TelemetryUnavailableError, match="required"):
        required.detect(frame)
