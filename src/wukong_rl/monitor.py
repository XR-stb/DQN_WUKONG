from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .types import CombatToken, MovementToken


def _recent_jsonl(path: Path, maximum_lines: int) -> list[dict[str, Any]]:
    if maximum_lines <= 0 or not path.exists():
        return []
    block_size = 64 * 1024
    chunks: list[bytes] = []
    newline_count = 0
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        position = stream.tell()
        while position > 0 and newline_count <= maximum_lines:
            size = min(block_size, position)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    lines = b"".join(reversed(chunks)).splitlines()[-maximum_lines:]
    events: list[dict[str, Any]] = []
    for raw in lines:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _action_distribution(
    commands: np.ndarray,
    enum_type,
    column: int,
) -> list[tuple[str, float]]:
    if commands.size == 0:
        return []
    counts = Counter(np.asarray(commands[:, column], dtype=np.int64).tolist())
    total = max(sum(counts.values()), 1)
    result: list[tuple[str, float]] = []
    for index, count in counts.most_common(4):
        try:
            name = enum_type(index).name
        except ValueError:
            name = f"UNKNOWN_{index}"
        result.append((name, count / total))
    return result


def _replay_snapshot(directory: Path, episode_window: int) -> dict[str, Any]:
    required = (
        "global_ids.npy",
        "episode_ids.npy",
        "step_ids.npy",
        "features.npy",
        "confidence.npy",
        "actions.npy",
        "rewards.npy",
        "terminated.npy",
        "truncated.npy",
    )
    if not all((directory / name).exists() for name in required):
        return {"available": False}

    global_ids = np.load(directory / "global_ids.npy", mmap_mode="r")
    latest_global_id = int(np.asarray(global_ids).max(initial=-1))
    if latest_global_id < 0:
        return {"available": False}
    capacity = len(global_ids)
    history = min(capacity, max(20_000, episode_window * 2_000))
    first_global_id = max(0, latest_global_id - history + 1)
    logical_ids = np.arange(first_global_id, latest_global_id + 1, dtype=np.int64)
    slots = logical_ids % capacity
    valid = np.asarray(global_ids[slots]) == logical_ids
    logical_ids = logical_ids[valid]
    slots = slots[valid]
    if slots.size == 0:
        return {"available": False}

    episode_ids = np.asarray(np.load(directory / "episode_ids.npy", mmap_mode="r")[slots])
    step_ids = np.asarray(np.load(directory / "step_ids.npy", mmap_mode="r")[slots])
    features = np.asarray(np.load(directory / "features.npy", mmap_mode="r")[slots])
    confidence = np.asarray(np.load(directory / "confidence.npy", mmap_mode="r")[slots])
    actions = np.asarray(np.load(directory / "actions.npy", mmap_mode="r")[slots])
    rewards = np.asarray(np.load(directory / "rewards.npy", mmap_mode="r")[slots])
    terminated = np.asarray(np.load(directory / "terminated.npy", mmap_mode="r")[slots])
    truncated = np.asarray(np.load(directory / "truncated.npy", mmap_mode="r")[slots])

    episodes: list[dict[str, Any]] = []
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        order = np.argsort(step_ids[mask])
        episode_features = features[mask][order]
        episode_confidence = confidence[mask][order]
        episode_rewards = rewards[mask][order]
        episode_terminated = terminated[mask][order]
        episode_truncated = truncated[mask][order]
        complete = bool(np.any(episode_terminated | episode_truncated))
        reliable_boss = episode_confidence[:, 1] >= 0.55
        boss_health = (
            float(np.min(episode_features[reliable_boss, 1]) * 100.0)
            if np.any(reliable_boss)
            else math.nan
        )
        self_health = float(episode_features[-1, 0] * 100.0)
        state = "running"
        if complete:
            if boss_health <= 3.0 and self_health > 3.0:
                state = "won"
            elif self_health <= 3.0:
                state = "lost"
            elif bool(np.any(episode_truncated)):
                state = "truncated"
            else:
                state = "terminated"
        episodes.append(
            {
                "episode_id": int(episode_id),
                "steps": int(mask.sum()),
                "reward": float(episode_rewards.sum()),
                "boss_health": boss_health,
                "self_health": self_health,
                "damage_dealt": 100.0 - boss_health,
                "state": state,
                "complete": complete,
            }
        )

    complete_episodes = [episode for episode in episodes if episode["complete"]]
    current = episodes[-1]
    recent_actions = actions[-min(len(actions), 5_000) :]
    recent_confidence = confidence[-min(len(confidence), 5_000) :]
    return {
        "available": True,
        "next_global_id": latest_global_id + 1,
        "current": current,
        "episodes": complete_episodes[-episode_window * 2 :],
        "movement": _action_distribution(recent_actions, MovementToken, 0),
        "combat": _action_distribution(recent_actions, CombatToken, 1),
        "boss_invalid_rate": float(np.mean(recent_confidence[:, 1] < 0.55)),
        "self_invalid_rate": float(np.mean(recent_confidence[:, 0] < 0.55)),
    }


def _episode_summary(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        return {}
    damages = np.asarray([event["damage_dealt"] for event in episodes], dtype=np.float64)
    rewards = np.asarray([event["reward"] for event in episodes], dtype=np.float64)
    steps = np.asarray([event["steps"] for event in episodes], dtype=np.float64)
    return {
        "count": float(len(episodes)),
        "wins": float(sum(event["state"] == "won" for event in episodes)),
        "damage_mean": float(np.nanmean(damages)),
        "damage_median": float(np.nanmedian(damages)),
        "damage_best": float(np.nanmax(damages)),
        "reward_mean": float(np.mean(rewards)),
        "steps_mean": float(np.mean(steps)),
    }


def _latest_counter_session(
    events: list[dict[str, Any]], counter: str
) -> list[dict[str, Any]]:
    """Drop metrics from earlier runs when append-only logs are reused."""

    if len(events) < 2:
        return events
    start = 0
    for index in range(1, len(events)):
        before = events[index - 1].get(counter)
        after = events[index].get(counter)
        if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
            continue
        delta = float(after) - float(before)
        if delta < 0 or delta > 100:
            start = index
    return events[start:]


def load_snapshot(
    metrics_directory: str | Path,
    replay_directory: str | Path,
    episode_window: int = 10,
    *,
    include_series: bool = False,
) -> dict[str, Any]:
    if episode_window < 2:
        raise ValueError("episode_window must be at least 2")
    metrics_directory = Path(metrics_directory)
    actor_events = _recent_jsonl(metrics_directory / "actor.jsonl", 12_000)
    learner_events = _recent_jsonl(metrics_directory / "train.jsonl", 2_000)
    restart_events = _recent_jsonl(
        metrics_directory / "restart-events.jsonl", 1
    )
    actor_steps = _latest_counter_session(
        [event for event in actor_events if event.get("kind") == "actor_step"],
        "environment_steps",
    )
    metric_episodes = [event for event in actor_events if event.get("kind") == "episode"]
    learner_steps = _latest_counter_session(
        [event for event in learner_events if event.get("kind") == "learner"],
        "learner_steps",
    )
    if actor_steps:
        session_started = float(actor_steps[0].get("timestamp", 0.0))
        metric_episodes = [
            event
            for event in metric_episodes
            if float(event.get("timestamp", 0.0)) >= session_started
        ]
    now = time.time()
    latest_restart = restart_events[-1] if restart_events else {}
    restart_timestamp = float(latest_restart.get("timestamp_unix_ns", 0)) / 1.0e9
    restart_age = now - restart_timestamp if restart_timestamp else math.inf
    actor_age = now - float(actor_steps[-1]["timestamp"]) if actor_steps else math.inf
    learner_age = now - float(learner_steps[-1]["timestamp"]) if learner_steps else math.inf
    replay = _replay_snapshot(Path(replay_directory), episode_window)

    episodes = list(replay.get("episodes", []))
    recent = episodes[-episode_window:]
    previous = episodes[-episode_window * 2 : -episode_window]
    latest_actor = actor_steps[-1] if actor_steps else {}
    actor_window = actor_steps[-min(len(actor_steps), 5_000) :]
    policy_summary: dict[str, float] = {}
    if actor_window:
        policy_summary["intervention_rate"] = float(
            np.mean([bool(event.get("policy_intervention")) for event in actor_window])
        )
        timestamps = np.asarray(
            [
                float(event.get("actor_timestamp", event.get("timestamp", 0.0)))
                for event in actor_window
            ],
            dtype=np.float64,
        )
        intervals = np.diff(timestamps)
        active_intervals = intervals[(intervals > 0.0) & (intervals <= 0.5)]
        if active_intervals.size:
            policy_summary["control_fps"] = float(1.0 / active_intervals.mean())
        for key in ("potion_allowed", "skill4_allowed", "transformation_active"):
            available = [event[key] for event in actor_window if key in event]
            if available:
                policy_summary[f"{key}_rate"] = float(
                    np.mean([bool(value) for value in available])
                )
    latest_learner = learner_steps[-1] if learner_steps else {}
    learner_window = learner_steps[-min(len(learner_steps), 100) :]
    learner_summary: dict[str, Any] = {}
    if learner_window:
        learner_summary = {
            "loss": float(np.mean([event["loss"] for event in learner_window])),
            "td_loss": float(np.mean([event["td_loss"] for event in learner_window])),
            "demo_loss": float(np.mean([event["demo_loss"] for event in learner_window])),
            "mean_q": float(np.mean([event["mean_q"] for event in learner_window])),
            "mean_target": float(
                np.mean([event["mean_target"] for event in learner_window])
            ),
            "gradient_norm": float(
                np.mean([event["gradient_norm"] for event in learner_window])
            ),
            "latency_ms": float(
                np.mean([event["learner_latency_ms"] for event in learner_window])
            ),
            "updates_per_second": (
                float(learner_window[-1]["learner_steps"] - learner_window[0]["learner_steps"])
                / max(
                    float(learner_window[-1]["timestamp"] - learner_window[0]["timestamp"]),
                    1.0e-6,
                )
            ),
        }
    latest_episode_metric = metric_episodes[-1] if metric_episodes else {}
    snapshot = {
        "timestamp": now,
        "episode_window": episode_window,
        "actor_age": actor_age,
        "learner_age": learner_age,
        "restart_age": restart_age,
        "latest_restart": latest_restart,
        "actor": latest_actor,
        "policy_summary": policy_summary,
        "learner": latest_learner,
        "learner_summary": learner_summary,
        "latest_episode_metric": latest_episode_metric,
        "replay": replay,
        "recent": _episode_summary(recent),
        "previous": _episode_summary(previous),
    }
    if include_series:
        # The graphical dashboard reuses the already parsed bounded tails.
        # Keeping them opt-in avoids enlarging the terminal monitor snapshot.
        snapshot["actor_steps"] = actor_steps
        snapshot["learner_steps"] = learner_steps
        snapshot["metric_episodes"] = metric_episodes
    return snapshot


def snapshot_status(snapshot: dict[str, Any]) -> str:
    """Return the lifecycle state shared by text and graphical monitors."""

    actor_age = float(snapshot.get("actor_age", math.inf))
    learner_age = float(snapshot.get("learner_age", math.inf))
    restart_age = float(snapshot.get("restart_age", math.inf))
    replay = snapshot.get("replay", {})
    if actor_age <= 10.0 and learner_age <= 10.0:
        return "LIVE"
    if restart_age <= 5.0:
        return "RESTARTING"
    if not replay.get("available"):
        return "WAITING"
    if actor_age > 10.0 and learner_age > 10.0:
        return "STOPPED"
    return "DEGRADED"


def _format_distribution(values: list[tuple[str, float]]) -> str:
    if not values:
        return "-"
    return "  ".join(f"{name} {ratio:.1%}" for name, ratio in values)


def _format_flag(event: dict[str, Any], key: str) -> str:
    return "-" if key not in event else str(int(bool(event[key])))


def render_snapshot(snapshot: dict[str, Any]) -> str:
    replay = snapshot["replay"]
    recent = snapshot["recent"]
    previous = snapshot["previous"]
    actor = snapshot["actor"]
    policy_summary = snapshot.get("policy_summary", {})
    learner = snapshot["learner"]
    learner_summary = snapshot["learner_summary"]
    actor_age = snapshot["actor_age"]
    learner_age = snapshot["learner_age"]
    restart_age = snapshot.get("restart_age", math.inf)
    latest_restart = snapshot.get("latest_restart", {})
    status = snapshot_status(snapshot)
    lines = [
        "Wukong RL 训练监控",
        f"状态: {status}  刷新时间: "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snapshot['timestamp']))}",
    ]
    if status == "WAITING":
        lines.extend(
            [
                "",
                "当前 v4 回放尚未产生 step。若训练刚启动，请等待 Learner 重建示范回放；",
                "若训练终端没有运行，请先启动 train。下方带 age 的内容是历史最后值。",
            ]
        )
    elif status == "RESTARTING":
        lines.extend(
            [
                "",
                "正在等待复战或可靠战斗画面；此阶段 Actor/Learner 暂停更新属于正常现象。",
                f"最近复战日志: {latest_restart.get('message', '-')}",
            ]
        )
    if replay.get("available"):
        current = replay["current"]
        current_label = "当前回合" if not current.get("complete") else "最新回合（已结束）"
        lines.extend(
            [
                "",
                current_label,
                f"  episode={current['episode_id']} step={current['steps']} "
                f"reward={current['reward']:.3f} boss_hp={current['boss_health']:.1f}% "
                f"self_hp={current['self_health']:.1f}% replay_steps={replay['next_global_id']}",
            ]
        )
    if recent:
        count = int(recent["count"])
        lines.extend(
            [
                "",
                f"最近 {count} 个完整回合",
                f"  wins={int(recent['wins'])}/{count} "
                f"boss_damage mean={recent['damage_mean']:.1f}% "
                f"median={recent['damage_median']:.1f}% best={recent['damage_best']:.1f}%",
                f"  reward mean={recent['reward_mean']:.3f}  "
                f"survival mean={recent['steps_mean']:.0f} steps",
            ]
        )
        if previous:
            damage_delta = recent["damage_mean"] - previous["damage_mean"]
            median_delta = recent["damage_median"] - previous["damage_median"]
            reward_delta = recent["reward_mean"] - previous["reward_mean"]
            survival_delta = recent["steps_mean"] - previous["steps_mean"]
            lines.append(
                f"  对比前窗: damage_mean={damage_delta:+.1f}% "
                f"median={median_delta:+.1f}% reward={reward_delta:+.3f} "
                f"survival={survival_delta:+.0f} steps"
            )
    lines.extend(
        [
            "",
            "策略分布（最近最多 5000 steps）",
            f"  movement: {_format_distribution(replay.get('movement', []))}",
            f"  combat:   {_format_distribution(replay.get('combat', []))}",
        ]
    )
    if learner_summary:
        lines.extend(
            [
                "",
                (
                    "Learner（最近 100 updates）"
                    if learner_age <= 10.0
                    else "Learner（历史最后值，非实时）"
                ),
                f"  step={learner.get('learner_steps', '-')} age={learner_age:.1f}s "
                f"loss={learner_summary['loss']:.4f} td={learner_summary['td_loss']:.4f} "
                f"demo={learner_summary['demo_loss']:.4f}",
                f"  Q={learner_summary['mean_q']:.3f} target={learner_summary['mean_target']:.3f} "
                f"grad={learner_summary['gradient_norm']:.3f} "
                f"speed={learner_summary['updates_per_second']:.2f} update/s "
                f"latency={learner_summary['latency_ms']:.1f}ms",
            ]
        )
    if actor:
        lines.extend(
            [
                "",
                "Actor / 感知" if actor_age <= 10.0 else "Actor / 感知（历史最后值，非实时）",
                f"  metrics_age={actor_age:.1f}s "
                f"control_fps={policy_summary.get('control_fps', math.nan):.2f} "
                f"run_avg_fps={actor.get('environment_fps', math.nan):.2f} "
                f"deadline_miss={actor.get('deadline_miss_rate', math.nan):.1%} "
                f"actor_latency={actor.get('actor_latency_ms', math.nan):.1f}ms",
                f"  confidence={actor.get('detection_confidence', math.nan):.3f} "
                f"invalid={actor.get('invalid_observation_rate', math.nan):.1%} "
                f"telemetry_age={actor.get('telemetry_age_ms', math.nan):.1f}ms",
                f"  Q可喝药={_format_flag(actor, 'potion_allowed')} "
                f"skill4可用={_format_flag(actor, 'skill4_allowed')} "
                f"变身中={_format_flag(actor, 'transformation_active')}",
            ]
        )
        if actor.get("movement_top") or actor.get("combat_top"):
            lines.append(
                f"  Q首选 movement={actor.get('movement_top', ['-'])} "
                f"margin={actor.get('movement_margin', math.nan)}  "
                f"combat={actor.get('combat_top', ['-'])} "
                f"margin={actor.get('combat_margin', math.nan)}"
            )
    if policy_summary:
        availability = []
        for label, key in (
            ("Q开放", "potion_allowed_rate"),
            ("skill4开放", "skill4_allowed_rate"),
            ("变身", "transformation_active_rate"),
        ):
            if key in policy_summary:
                availability.append(f"{label}={policy_summary[key]:.1%}")
        lines.append(
            f"  最近动作干预率={policy_summary['intervention_rate']:.1%}"
            + ("  " + "  ".join(availability) if availability else "")
        )
    episode_metric = snapshot["latest_episode_metric"]
    if "reward_boss_damage" in episode_metric:
        lines.extend(
            [
                "",
                "最近一局奖励分解",
                f"  boss={episode_metric['reward_boss_damage']:+.3f} "
                f"self={episode_metric['reward_self_damage']:+.3f} "
                f"tick={episode_metric['reward_tick']:+.3f} "
                f"terminal={episode_metric['reward_terminal']:+.3f} "
                f"clip={episode_metric.get('reward_clipping', 0.0):+.3f}",
            ]
        )

    warnings: list[str] = []
    if actor_age > 10.0 and status != "RESTARTING":
        warnings.append(f"Actor metrics 已停更 {actor_age:.0f}s，当前值改由 replay 恢复")
    if learner_age > 10.0 and status != "RESTARTING":
        warnings.append(f"Learner metrics 已停更 {learner_age:.0f}s，loss/Q 仅是最后已知值")
    if recent and recent["wins"] == 0:
        warnings.append("最近窗口没有胜局")
    if recent and previous and (
        recent["wins"] <= previous["wins"]
        and (
            recent["damage_median"] <= previous["damage_median"] + 2.0
            or recent["reward_mean"] <= previous["reward_mean"]
        )
    ):
        warnings.append("Boss 伤害、回报和胜率尚未形成一致的上升趋势")
    movement = replay.get("movement", [])
    if movement and movement[0][1] >= 0.28:
        warnings.append(f"移动策略偏向 {movement[0][0]} ({movement[0][1]:.1%})")
    if replay.get("boss_invalid_rate", 0.0) >= 0.05:
        warnings.append(f"Boss 血量低置信度比例 {replay['boss_invalid_rate']:.1%}")
    warnings.append("在线带探索曲线只用于早期预警，最终收敛必须用 exploration=0 冻结评测")
    lines.extend(["", "诊断"] + [f"  ! {warning}" for warning in warnings])
    return "\n".join(lines)


def run_monitor(
    metrics_directory: str | Path,
    replay_directory: str | Path,
    *,
    refresh_seconds: float = 5.0,
    episode_window: int = 10,
    once: bool = False,
) -> None:
    if refresh_seconds < 1.0:
        raise ValueError("refresh_seconds must be at least 1")
    try:
        while True:
            snapshot = load_snapshot(metrics_directory, replay_directory, episode_window)
            rendered = render_snapshot(snapshot)
            if once:
                print(rendered, flush=True)
                return
            print("\x1b[2J\x1b[H" + rendered, end="\n", flush=True)
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        print("\n[monitor] 监控已停止；训练进程不受影响。", flush=True)
