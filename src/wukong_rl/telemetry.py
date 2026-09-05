from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable, Mapping

import numpy as np

from .config import TelemetryConfig
from .types import FieldMeasurement


SCHEMA_VERSION = 1


class TelemetryUnavailableError(RuntimeError):
    """Raised when telemetry is configured as required but is not fresh."""


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer or null")
    return int(value)


def _optional_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean or null")
    return value


@dataclass(slots=True, frozen=True)
class EntityTelemetry:
    valid: bool
    res_id: int | None = None
    unique_id: int | None = None
    hp: float | None = None
    hp_max: float | None = None
    mp: float | None = None
    mp_max: float | None = None
    stamina: float | None = None
    stamina_max: float | None = None
    focus_level: float | None = None
    focus_level_max: float | None = None
    focus_value: float | None = None
    focus_value_max: float | None = None
    fabao_energy: float | None = None
    fabao_energy_max: float | None = None
    vigor_energy: float | None = None
    vigor_energy_max: float | None = None
    dead: bool | None = None
    in_battle: bool | None = None

    @classmethod
    def from_mapping(cls, payload: Any, field_name: str) -> EntityTelemetry:
        if payload is None:
            return cls(valid=False)
        if not isinstance(payload, Mapping):
            raise ValueError(f"{field_name} must be an object or null")
        valid = payload.get("valid", False)
        if not isinstance(valid, bool):
            raise ValueError(f"{field_name}.valid must be a boolean")
        return cls(
            valid=valid,
            res_id=_optional_int(payload.get("res_id"), f"{field_name}.res_id"),
            unique_id=_optional_int(payload.get("unique_id"), f"{field_name}.unique_id"),
            hp=_optional_float(payload.get("hp"), f"{field_name}.hp"),
            hp_max=_optional_float(payload.get("hp_max"), f"{field_name}.hp_max"),
            mp=_optional_float(payload.get("mp"), f"{field_name}.mp"),
            mp_max=_optional_float(payload.get("mp_max"), f"{field_name}.mp_max"),
            stamina=_optional_float(payload.get("stamina"), f"{field_name}.stamina"),
            stamina_max=_optional_float(
                payload.get("stamina_max"), f"{field_name}.stamina_max"
            ),
            focus_level=_optional_float(
                payload.get("focus_level"), f"{field_name}.focus_level"
            ),
            focus_level_max=_optional_float(
                payload.get("focus_level_max"), f"{field_name}.focus_level_max"
            ),
            focus_value=_optional_float(
                payload.get("focus_value"), f"{field_name}.focus_value"
            ),
            focus_value_max=_optional_float(
                payload.get("focus_value_max"), f"{field_name}.focus_value_max"
            ),
            fabao_energy=_optional_float(
                payload.get("fabao_energy"), f"{field_name}.fabao_energy"
            ),
            fabao_energy_max=_optional_float(
                payload.get("fabao_energy_max"), f"{field_name}.fabao_energy_max"
            ),
            vigor_energy=_optional_float(
                payload.get("vigor_energy"), f"{field_name}.vigor_energy"
            ),
            vigor_energy_max=_optional_float(
                payload.get("vigor_energy_max"), f"{field_name}.vigor_energy_max"
            ),
            dead=_optional_bool(payload.get("dead"), f"{field_name}.dead"),
            in_battle=_optional_bool(
                payload.get("in_battle"), f"{field_name}.in_battle"
            ),
        )


@dataclass(slots=True, frozen=True)
class SkillTelemetry:
    slot: int
    skill_id: int
    ready: bool | None
    active: bool | None

    @classmethod
    def from_mapping(cls, payload: Any) -> SkillTelemetry:
        if not isinstance(payload, Mapping):
            raise ValueError("skills entries must be objects")
        slot = _optional_int(payload.get("slot"), "skills.slot")
        skill_id = _optional_int(payload.get("skill_id"), "skills.skill_id")
        if slot is None or not 0 <= slot < 4:
            raise ValueError("skills.slot must be within [0, 3]")
        if skill_id is None or skill_id < 0:
            raise ValueError("skills.skill_id must be non-negative")
        return cls(
            slot=slot,
            skill_id=skill_id,
            ready=_optional_bool(payload.get("ready"), "skills.ready"),
            active=_optional_bool(payload.get("active"), "skills.active"),
        )


@dataclass(slots=True, frozen=True)
class TelemetrySnapshot:
    schema_version: int
    sequence: int
    emitted_unix_ns: int
    player: EntityTelemetry
    target: EntityTelemetry
    skills: tuple[SkillTelemetry, ...]
    last_skill_id: int | None = None
    received_monotonic: float = 0.0

    @classmethod
    def from_json(
        cls, payload: str | bytes, *, received_monotonic: float = 0.0
    ) -> TelemetrySnapshot:
        try:
            decoded = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("telemetry packet is not valid JSON") from error
        if not isinstance(decoded, Mapping):
            raise ValueError("telemetry packet must be a JSON object")
        schema_version = _optional_int(decoded.get("schema_version"), "schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported telemetry schema {schema_version!r}; expected {SCHEMA_VERSION}"
            )
        sequence = _optional_int(decoded.get("sequence"), "sequence")
        emitted_unix_ns = _optional_int(decoded.get("emitted_unix_ns"), "emitted_unix_ns")
        if sequence is None or sequence < 0:
            raise ValueError("sequence must be a non-negative integer")
        if emitted_unix_ns is None or emitted_unix_ns < 0:
            raise ValueError("emitted_unix_ns must be a non-negative integer")
        raw_skills = decoded.get("skills", [])
        if not isinstance(raw_skills, list) or len(raw_skills) > 4:
            raise ValueError("skills must be an array with at most four entries")
        skills = tuple(SkillTelemetry.from_mapping(item) for item in raw_skills)
        if len({skill.slot for skill in skills}) != len(skills):
            raise ValueError("skills contains duplicate slots")
        last_skill_id = _optional_int(decoded.get("last_skill_id"), "last_skill_id")
        if last_skill_id is not None and last_skill_id < 0:
            raise ValueError("last_skill_id must be non-negative or null")
        return cls(
            schema_version=schema_version,
            sequence=sequence,
            emitted_unix_ns=emitted_unix_ns,
            player=EntityTelemetry.from_mapping(decoded.get("player"), "player"),
            target=EntityTelemetry.from_mapping(decoded.get("target"), "target"),
            skills=skills,
            last_skill_id=last_skill_id,
            received_monotonic=float(received_monotonic),
        )


@dataclass(slots=True, frozen=True)
class TelemetryStatus:
    running: bool
    connected: bool
    fresh: bool
    sequence: int | None
    age_ms: float | None
    valid_packets: int
    invalid_packets: int
    last_error: str | None


PipeOpener = Callable[[Path], BinaryIO]


def _open_pipe(path: Path) -> BinaryIO:
    return path.open("rb", buffering=0)


class NamedPipeTelemetryClient:
    """Non-blocking latest-value client for the local read-only mod pipe."""

    def __init__(
        self,
        config: TelemetryConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        pipe_opener: PipeOpener = _open_pipe,
    ) -> None:
        self.config = config
        self.clock = clock
        self.pipe_path = Path(rf"\\.\pipe\{config.pipe_name}")
        self._pipe_opener = pipe_opener
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._handle_lock = threading.Lock()
        self._handle: BinaryIO | None = None
        self._thread: threading.Thread | None = None
        self._latest: TelemetrySnapshot | None = None
        self._connected = False
        self._valid_packets = 0
        self._invalid_packets = 0
        self._last_error: str | None = None

    def start(self) -> None:
        if self.config.mode == "off" or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="wukong-telemetry", daemon=True
        )
        self._thread.start()

    def _set_connection(self, connected: bool, error: str | None = None) -> None:
        with self._lock:
            self._connected = connected
            self._last_error = error

    def ingest_line(self, line: str | bytes) -> TelemetrySnapshot:
        received = self.clock()
        try:
            snapshot = TelemetrySnapshot.from_json(
                line, received_monotonic=received
            )
        except ValueError:
            with self._lock:
                self._invalid_packets += 1
            raise
        with self._lock:
            self._latest = snapshot
            self._valid_packets += 1
            self._last_error = None
        return snapshot

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                handle = self._pipe_opener(self.pipe_path)
                with self._handle_lock:
                    self._handle = handle
                self._set_connection(True)
                while not self._stop.is_set():
                    line = handle.readline()
                    if not line:
                        raise OSError("telemetry pipe closed by server")
                    try:
                        self.ingest_line(line)
                    except ValueError as error:
                        self._set_connection(True, str(error))
            except (OSError, EOFError, ValueError) as error:
                self._set_connection(False, str(error))
            finally:
                with self._handle_lock:
                    handle = self._handle
                    self._handle = None
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass
            self._stop.wait(self.config.reconnect_seconds)

    def latest(self, now: float | None = None) -> TelemetrySnapshot | None:
        timestamp = self.clock() if now is None else now
        with self._lock:
            snapshot = self._latest
        if snapshot is None:
            return None
        if timestamp - snapshot.received_monotonic > self.config.max_age_seconds:
            return None
        return snapshot

    def status(self, now: float | None = None) -> TelemetryStatus:
        timestamp = self.clock() if now is None else now
        with self._lock:
            snapshot = self._latest
            connected = self._connected
            valid_packets = self._valid_packets
            invalid_packets = self._invalid_packets
            last_error = self._last_error
        age_ms = (
            None
            if snapshot is None
            else max(0.0, (timestamp - snapshot.received_monotonic) * 1000.0)
        )
        return TelemetryStatus(
            running=bool(self._thread and self._thread.is_alive()),
            connected=connected,
            fresh=bool(age_ms is not None and age_ms <= self.config.max_age_seconds * 1000.0),
            sequence=None if snapshot is None else snapshot.sequence,
            age_ms=age_ms,
            valid_packets=valid_packets,
            invalid_packets=invalid_packets,
            last_error=last_error,
        )

    def close(self) -> None:
        self._stop.set()
        with self._handle_lock:
            handle = self._handle
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)


def _percentage(current: float | None, maximum: float | None) -> FieldMeasurement | None:
    if current is None or maximum is None or maximum <= 0:
        return None
    value = float(np.clip(100.0 * current / maximum, 0.0, 100.0))
    return FieldMeasurement(value=value, confidence=1.0, age=0, valid=True)


def snapshot_summary(snapshot: TelemetrySnapshot) -> dict[str, Any]:
    """Small, stable probe payload; avoids dumping every internal field."""

    player_hp = _percentage(snapshot.player.hp, snapshot.player.hp_max)
    target_hp = _percentage(snapshot.target.hp, snapshot.target.hp_max)
    return {
        "sequence": snapshot.sequence,
        "player_hp_percent": None if player_hp is None else round(player_hp.value, 3),
        "player_mp_percent": (
            None
            if (value := _percentage(snapshot.player.mp, snapshot.player.mp_max)) is None
            else round(value.value, 3)
        ),
        "stamina_percent": (
            None
            if (
                value := _percentage(
                    snapshot.player.stamina, snapshot.player.stamina_max
                )
            )
            is None
            else round(value.value, 3)
        ),
        "focus_level": snapshot.player.focus_level,
        "player_dead": snapshot.player.dead,
        "in_battle": snapshot.player.in_battle,
        "target_valid": snapshot.target.valid,
        "target_res_id": snapshot.target.res_id,
        "target_unique_id": snapshot.target.unique_id,
        "target_hp_percent": None if target_hp is None else round(target_hp.value, 3),
        "target_dead": snapshot.target.dead,
        "skills": [
            {
                "slot": skill.slot,
                "skill_id": skill.skill_id,
                "ready": skill.ready,
                "active": skill.active,
            }
            for skill in snapshot.skills
        ],
        "last_skill_id": snapshot.last_skill_id,
    }


class HybridPerception:
    """Fuse exact read-only telemetry into the existing visual observation schema."""

    def __init__(self, screen, client: NamedPipeTelemetryClient, config: TelemetryConfig) -> None:
        self.screen = screen
        self.client = client
        self.config = config
        self.last_sources: dict[str, str] = {}
        self.client.start()

    def reset(self) -> None:
        self.screen.reset()

    def set_episode_active(self, active: bool) -> None:
        setter = getattr(self.screen, "set_episode_active", None)
        if setter is not None:
            setter(active)

    def _put_percentage(
        self,
        measurements: dict[str, FieldMeasurement],
        key: str,
        current: float | None,
        maximum: float | None,
    ) -> None:
        measurement = _percentage(current, maximum)
        if measurement is not None:
            measurements[key] = measurement
            self.last_sources[key] = "telemetry"

    def detect(self, frame: np.ndarray) -> dict[str, FieldMeasurement]:
        measurements = dict(self.screen.detect(frame))
        self.last_sources = {key: "screen" for key in measurements}
        snapshot = self.client.latest()
        if snapshot is None:
            if self.config.mode == "required":
                raise TelemetryUnavailableError(
                    "fresh game telemetry is required but the named pipe is unavailable"
                )
            return measurements

        player = snapshot.player
        if player.valid:
            self._put_percentage(measurements, "self_blood", player.hp, player.hp_max)
            self._put_percentage(measurements, "self_magic", player.mp, player.mp_max)
            self._put_percentage(
                measurements, "self_energy", player.stamina, player.stamina_max
            )
            if player.focus_level is not None:
                for index, key in enumerate(("gunshi1", "gunshi2", "gunshi3"), start=1):
                    measurements[key] = FieldMeasurement(
                        value=float(player.focus_level >= index),
                        confidence=1.0,
                        age=0,
                        valid=True,
                    )
                    self.last_sources[key] = "telemetry"

        target = snapshot.target
        target_allowed = not self.config.accepted_boss_res_ids or (
            target.res_id in self.config.accepted_boss_res_ids
        )
        if target.valid and target_allowed:
            self._put_percentage(measurements, "boss_blood", target.hp, target.hp_max)

        skills_by_slot = {skill.slot: skill for skill in snapshot.skills}
        for slot, expected_id in enumerate(self.config.skill_ids):
            skill = skills_by_slot.get(slot)
            if (
                expected_id <= 0
                or skill is None
                or skill.skill_id != expected_id
                or skill.ready is None
            ):
                continue
            key = f"skill_{slot + 1}"
            measurements[key] = FieldMeasurement(
                value=float(skill.ready), confidence=1.0, age=0, valid=True
            )
            self.last_sources[key] = "telemetry"
        return measurements

    def close(self) -> None:
        self.client.close()


def build_perception(config):
    """Build the configured live detector without creating an input backend."""

    from .perception import ScreenPerception

    screen = ScreenPerception(
        config.perception, config.capture.width, config.capture.height
    )
    if config.telemetry.mode == "off":
        return screen
    return HybridPerception(
        screen,
        NamedPipeTelemetryClient(config.telemetry),
        config.telemetry,
    )
