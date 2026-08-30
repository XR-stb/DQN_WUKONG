from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .actions import build_action_mask
from .agent import R2D3Agent
from .capture import LegacyScreenSource
from .config import PipelineConfig
from .perception import ScreenPerception
from .replay import ReplayBatch
from .types import ActionToken, HUD_KEYS, measurements_to_arrays


def calibrate(config: PipelineConfig, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = LegacyScreenSource(config.capture)
    perception = ScreenPerception(config.perception, config.capture.width, config.capture.height)
    source.start()
    try:
        measurements = None
        frame = None
        for _ in range(max(config.perception.confirm_frames, 3)):
            frame = source.read()
            measurements = perception.detect(frame)
            time.sleep(0.05)
        annotated = frame.copy()
        for name, detector in perception.detectors.items():
            x1, y1, x2, y2 = detector.coordinates
            measurement = measurements[name]
            color = (0, 220, 0) if measurement.valid else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                annotated,
                f"{name}:{measurement.value:.1f}/{measurement.confidence:.2f}",
                (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
        if not cv2.imwrite(str(output), annotated):
            raise RuntimeError(f"failed to write calibration image to {output}")
        report = {
            name: {
                "value": measurement.value,
                "confidence": measurement.confidence,
                "age": measurement.age,
                "valid": measurement.valid,
            }
            for name, measurement in measurements.items()
        }
        output.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return output
    finally:
        source.close()


def benchmark(config: PipelineConfig, iterations: int = 100, live_capture: bool = False) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config.model, device=device)
    frames = torch.randint(
        0,
        256,
        (1, 1, config.capture.observation_height, config.capture.observation_width, 3),
        dtype=torch.uint8,
        device=device,
    )
    features = torch.zeros((1, 1, len(HUD_KEYS)), device=device)
    confidence = torch.ones_like(features)
    previous_actions = torch.zeros((1, 1), dtype=torch.long, device=device)
    previous_rewards = torch.zeros((1, 1), device=device)
    state = agent.initial_state()
    for _ in range(10):
        with torch.no_grad():
            _, state = agent.online(
                frames, features, confidence, previous_actions, previous_rewards, state
            )
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        with torch.no_grad():
            _, state = agent.online(
                frames, features, confidence, previous_actions, previous_rewards, state
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)
    torch.set_num_threads(config.training.actor_cpu_threads)
    cpu_actor = R2D3Agent(
        len(HUD_KEYS), ActionToken.size(), config.model, device=torch.device("cpu")
    )
    cpu_frames = frames.cpu()
    cpu_features = features.cpu()
    cpu_confidence = confidence.cpu()
    cpu_previous_actions = previous_actions.cpu()
    cpu_previous_rewards = previous_rewards.cpu()
    cpu_state = cpu_actor.initial_state()
    cpu_timings: list[float] = []
    for _ in range(10):
        with torch.no_grad():
            _, cpu_state = cpu_actor.online(
                cpu_frames,
                cpu_features,
                cpu_confidence,
                cpu_previous_actions,
                cpu_previous_rewards,
                cpu_state,
            )
    for _ in range(iterations):
        started = time.perf_counter()
        with torch.no_grad():
            _, cpu_state = cpu_actor.online(
                cpu_frames,
                cpu_features,
                cpu_confidence,
                cpu_previous_actions,
                cpu_previous_rewards,
                cpu_state,
            )
        cpu_timings.append((time.perf_counter() - started) * 1000.0)
    capture_timings: list[float] = []
    if live_capture:
        source = LegacyScreenSource(config.capture)
        source.start()
        try:
            for _ in range(iterations):
                started = time.perf_counter()
                source.read()
                capture_timings.append((time.perf_counter() - started) * 1000.0)
        finally:
            source.close()
    perception = ScreenPerception(
        config.perception, config.capture.width, config.capture.height
    )
    benchmark_frame = np.random.default_rng(19).integers(
        0,
        256,
        (config.capture.height, config.capture.width, 3),
        dtype=np.uint8,
    )
    processing_timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        measurements = perception.detect(benchmark_frame)
        measurements_to_arrays(measurements)
        build_action_mask(measurements, config.environment.minimum_confidence)
        rgb = cv2.cvtColor(benchmark_frame, cv2.COLOR_BGR2RGB)
        cv2.resize(
            rgb,
            (config.capture.observation_width, config.capture.observation_height),
            interpolation=cv2.INTER_AREA,
        )
        processing_timings.append((time.perf_counter() - started) * 1000.0)
    model = config.model
    sequence_length = model.burn_in + model.unroll + model.n_step
    rng = np.random.default_rng(11)
    training_batch = ReplayBatch(
        frames=rng.integers(
            0,
            256,
            (
                model.batch_size,
                sequence_length + 1,
                config.capture.observation_height,
                config.capture.observation_width,
                3,
            ),
            dtype=np.uint8,
        ),
        features=rng.random(
            (model.batch_size, sequence_length + 1, len(HUD_KEYS)), dtype=np.float32
        ),
        confidence=np.ones(
            (model.batch_size, sequence_length + 1, len(HUD_KEYS)), dtype=np.float32
        ),
        action_masks=np.ones(
            (model.batch_size, sequence_length + 1, ActionToken.size()), dtype=np.bool_
        ),
        previous_actions=np.zeros((model.batch_size, sequence_length + 1), dtype=np.int64),
        previous_rewards=np.zeros((model.batch_size, sequence_length + 1), dtype=np.float32),
        actions=rng.integers(
            0, ActionToken.size(), (model.batch_size, sequence_length), dtype=np.int64
        ),
        rewards=rng.standard_normal((model.batch_size, sequence_length), dtype=np.float32),
        terminated=np.zeros((model.batch_size, sequence_length), dtype=np.bool_),
        truncated=np.zeros((model.batch_size, sequence_length), dtype=np.bool_),
        weights=np.ones(model.batch_size, dtype=np.float32),
        start_ids=np.arange(model.batch_size, dtype=np.int64),
        demonstrations=np.arange(model.batch_size) < round(model.batch_size * 0.25),
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()
    learner_metrics = agent.learn(training_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_ms = (time.perf_counter() - training_started) * 1000.0
    return {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in agent.online.parameters()),
        "inference_mean_ms": float(np.mean(timings)),
        "inference_p95_ms": float(np.percentile(timings, 95)),
        "actor_cpu_mean_ms": float(np.mean(cpu_timings)),
        "actor_cpu_p95_ms": float(np.percentile(cpu_timings, 95)),
        "capture_mean_ms": float(np.mean(capture_timings)) if capture_timings else None,
        "capture_p95_ms": float(np.percentile(capture_timings, 95)) if capture_timings else None,
        "observation_processing_mean_ms": float(np.mean(processing_timings)),
        "observation_processing_p95_ms": float(np.percentile(processing_timings, 95)),
        "learner_update_ms": training_ms,
        "learner_loss": learner_metrics.loss,
        "vram_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2)
        if device.type == "cuda"
        else 0.0,
    }
