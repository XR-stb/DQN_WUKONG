from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .agent import R2D3Agent


CHECKPOINT_SCHEMA_VERSION = 3


def checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema in {path}")
    return {
        key: payload.get(key)
        for key in (
            "schema_version",
            "config_hash",
            "learner_steps",
            "data_version",
            "extra",
        )
    }


def save_checkpoint(
    agent: R2D3Agent,
    path: str | Path,
    config_hash: str,
    extra: dict[str, Any] | None = None,
    normalization_state: dict[str, Any] | None = None,
    data_version: str | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_hash": config_hash,
        "learner_steps": agent.learner_steps,
        "online": agent.online.state_dict(),
        "target": agent.target.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "scaler": agent.scaler.state_dict(),
        "normalization_state": normalization_state or {},
        "data_version": data_version,
        "extra": extra or {},
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(
    agent: R2D3Agent,
    path: str | Path,
    *,
    expected_config_hash: str | None = None,
    expected_data_version: str | None = None,
    load_optimizer: bool = True,
) -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema in {path}")
    checkpoint_hash = payload.get("config_hash")
    if expected_config_hash and checkpoint_hash != expected_config_hash:
        raise ValueError(
            f"checkpoint/config mismatch: {checkpoint_hash} != {expected_config_hash}"
        )
    checkpoint_data_version = payload.get("data_version")
    if expected_data_version and checkpoint_data_version != expected_data_version:
        raise ValueError(
            "checkpoint/dataset mismatch: "
            f"{checkpoint_data_version} != {expected_data_version}"
        )
    agent.online.load_state_dict(payload["online"])
    agent.target.load_state_dict(payload.get("target", payload["online"]))
    if load_optimizer and "optimizer" in payload:
        agent.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler"):
            agent.scaler.load_state_dict(payload["scaler"])
    agent.learner_steps = int(payload.get("learner_steps", 0))
    return payload.get("extra", {})


def cpu_state_dict(agent: R2D3Agent) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in agent.online.state_dict().items()}
