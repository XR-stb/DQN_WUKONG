from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CaptureConfig:
    backend: str = "wgc"
    window_title: str = "b1  "
    width: int = 1280
    height: int = 720
    observation_width: int = 160
    observation_height: int = 90
    allow_dxcam_fallback: bool = False
    dxcam_osd_disabled: bool = False
    process_isolation: bool = True
    worker_fps: float = 30.0


@dataclass(slots=True)
class EnvironmentConfig:
    boss_id: str = "yinhu"
    control_hz: float = 8.0
    ready_timeout_seconds: float = 60.0
    episode_timeout_seconds: float = 300.0
    terminal_confirm_frames: int = 3
    minimum_confidence: float = 0.55
    restart_action: str = "FUZHAN_STAND_RESTART"


@dataclass(slots=True)
class RewardConfig:
    boss_damage_per_percent: float = 0.1
    self_damage_per_percent: float = -0.12
    tick_penalty: float = -0.001
    win_reward: float = 10.0
    loss_reward: float = -10.0
    nonterminal_clip: float = 2.0


@dataclass(slots=True)
class ModelConfig:
    hidden_size: int = 256
    gamma: float = 0.997
    n_step: int = 5
    burn_in: int = 8
    unroll: int = 32
    batch_size: int = 16
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    gradient_clip: float = 10.0
    target_update_interval: int = 2000
    demo_margin: float = 0.8
    demo_loss_weight: float = 1.0


@dataclass(slots=True)
class ReplayConfig:
    directory: str = "artifacts/replay"
    capacity_frames: int = 200_000
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    demo_ratio: float = 0.25
    minimum_sequences: int = 64


@dataclass(slots=True)
class TrainingConfig:
    checkpoint_directory: str = "artifacts/checkpoints"
    dataset_directory: str = "artifacts/datasets"
    metrics_directory: str = "artifacts/metrics"
    actor_epsilon_start: float = 0.2
    actor_epsilon_end: float = 0.02
    epsilon_decay_steps: int = 100_000
    weight_sync_seconds: float = 2.0
    updates_per_environment_step: float = 0.25
    checkpoint_interval_seconds: float = 300.0
    actor_cpu_threads: int = 4
    random_seed: int = 7


@dataclass(slots=True)
class PerceptionConfig:
    base_width: int = 1280
    base_height: int = 720
    confirm_frames: int = 3
    minimum_confidence: float = 0.55
    maximum_jump_percent: float = 35.0
    boss_increase_tolerance: float = 1.5
    opencv_threads: int = 1
    regions: dict[str, list[int]] = field(default_factory=dict)
    ranges: dict[str, list[float]] = field(default_factory=dict)


@dataclass(slots=True)
class MonitoringConfig:
    enabled: bool = True
    directory: str = "artifacts/profiles"
    resource_interval_seconds: float = 1.0
    summary_interval_seconds: float = 5.0
    queue_capacity: int = 2048
    quantile_window: int = 4096
    deadline_tolerance_ms: float = 5.0
    game_process_name: str = "b1-Win64-Shipping.exe"
    gpu_enabled: bool = True
    gpu_index: int = 0


@dataclass(slots=True)
class PipelineConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    def validate(self) -> None:
        if not all(math.isfinite(value) for value in (self.monitoring.resource_interval_seconds, self.monitoring.summary_interval_seconds, self.monitoring.deadline_tolerance_ms)):
            raise ValueError("monitoring timing settings must be finite")
        if self.monitoring.resource_interval_seconds < 0.5:
            raise ValueError("resource sampling interval must be at least 0.5 seconds")
        if self.monitoring.summary_interval_seconds < 1.0:
            raise ValueError("summary interval must be at least one second")
        if self.monitoring.queue_capacity < 8 or self.monitoring.quantile_window < 16:
            raise ValueError("monitoring queue/window is too small")
        if self.monitoring.deadline_tolerance_ms < 0 or self.monitoring.gpu_index < 0:
            raise ValueError("invalid monitoring deadline tolerance or GPU index")
        if self.capture.backend not in {"wgc", "dxcam", "array"}:
            raise ValueError(f"unsupported capture backend: {self.capture.backend}")
        if self.capture.width <= 0 or self.capture.height <= 0:
            raise ValueError("capture dimensions must be positive")
        if self.capture.observation_width <= 0 or self.capture.observation_height <= 0:
            raise ValueError("observation dimensions must be positive")
        if not math.isfinite(self.capture.worker_fps) or not 8.0 <= self.capture.worker_fps <= 60.0:
            raise ValueError("capture.worker_fps must be between 8 and 60")
        if self.capture.backend == "dxcam" and not self.capture.dxcam_osd_disabled:
            raise ValueError("dxcam requires dxcam_osd_disabled=true to prevent OSD contamination")
        if self.capture.allow_dxcam_fallback and not self.capture.dxcam_osd_disabled:
            raise ValueError("dxcam fallback requires dxcam_osd_disabled=true")
        if not 1.0 <= self.environment.control_hz <= 30.0:
            raise ValueError("control_hz must be between 1 and 30")
        if self.model.burn_in < 0 or self.model.unroll <= 0 or self.model.n_step <= 0:
            raise ValueError("invalid recurrent sequence lengths")
        if not 0.0 <= self.replay.demo_ratio <= 1.0:
            raise ValueError("demo_ratio must be within [0, 1]")
        if self.training.actor_cpu_threads <= 0:
            raise ValueError("actor_cpu_threads must be positive")
        if not 1 <= self.perception.opencv_threads <= 8:
            raise ValueError("perception.opencv_threads must be between 1 and 8")
        if self.replay.capacity_frames <= self.model.burn_in + self.model.unroll + self.model.n_step:
            raise ValueError("replay capacity is too small for one recurrent sequence")
        if not 0.0 < self.model.gamma <= 1.0:
            raise ValueError("gamma must be within (0, 1]")
        if self.model.batch_size <= 0 or self.model.learning_rate <= 0:
            raise ValueError("batch size and learning rate must be positive")
        required = {"self_blood", "boss_blood", "self_energy", "self_magic", "hulu"}
        missing = required.difference(self.perception.regions)
        if missing:
            raise ValueError(f"missing perception regions: {sorted(missing)}")
        for name, coordinates in self.perception.regions.items():
            if len(coordinates) != 4:
                raise ValueError(f"perception region {name} must contain four coordinates")
            x1, y1, x2, y2 = coordinates
            if not (0 <= x1 <= x2 < self.perception.base_width):
                raise ValueError(f"perception region {name} has invalid horizontal coordinates")
            if not (0 <= y1 <= y2 < self.perception.base_height):
                raise ValueError(f"perception region {name} has invalid vertical coordinates")

    def fingerprint(self) -> str:
        payload = asdict(self)
        # Observability does not change the policy/data semantics. Keep existing
        # checkpoint hashes valid when enabling a probe or changing its interval.
        payload.pop("monitoring", None)
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


def _construct(section_type: type, payload: dict[str, Any] | None):
    return section_type(**(payload or {}))


def load_config(path: str | Path = "config/rl_pipeline.yaml") -> PipelineConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    config = PipelineConfig(
        capture=_construct(CaptureConfig, raw.get("capture")),
        environment=_construct(EnvironmentConfig, raw.get("environment")),
        reward=_construct(RewardConfig, raw.get("reward")),
        perception=_construct(PerceptionConfig, raw.get("perception")),
        model=_construct(ModelConfig, raw.get("model")),
        replay=_construct(ReplayConfig, raw.get("replay")),
        training=_construct(TrainingConfig, raw.get("training")),
        monitoring=_construct(MonitoringConfig, raw.get("monitoring")),
    )
    config.validate()
    return config


def write_migrated_config(
    destination: str | Path,
    legacy_game: str | Path = "config/game_conf.yaml",
    legacy_models: str | Path = "config/models_conf.yaml",
) -> Path:
    """Create a new validated config while carrying over safe legacy settings."""
    warnings.warn(
        "legacy YAML is deprecated; review the generated rl_pipeline config before training",
        FutureWarning,
        stacklevel=2,
    )
    destination = Path(destination)
    with Path(legacy_game).open("r", encoding="utf-8") as handle:
        game = yaml.safe_load(handle) or {}
    with Path(legacy_models).open("r", encoding="utf-8") as handle:
        models = yaml.safe_load(handle) or {}
    config = PipelineConfig()
    window = game.get("game_window", {})
    config.capture.width = int(window.get("width", config.capture.width))
    config.capture.height = int(window.get("height", config.capture.height))
    config.environment.restart_action = models.get("training", {}).get(
        "restart_action", config.environment.restart_action
    )
    ui = game.get("ui_coordinates", {})
    active_boss = game.get("active_boss", "寅虎")
    config.perception.regions = {**ui}
    boss_region = game.get("boss_blood_presets", {}).get(active_boss)
    if boss_region:
        config.perception.regions["boss_blood"] = boss_region
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, allow_unicode=True, sort_keys=False)
    return destination
