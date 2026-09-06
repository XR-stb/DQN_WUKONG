from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .monitor import _recent_jsonl, load_snapshot, snapshot_status


_STATUS_COLORS = {
    "LIVE": "#35d07f",
    "RESTARTING": "#f6c85f",
    "WAITING": "#8b9eb7",
    "DEGRADED": "#ff9f43",
    "STOPPED": "#ef6262",
}


def load_events(path: str | Path, limit: int = 5000) -> list[dict[str, Any]]:
    """Read a bounded JSONL tail and ignore a partially written last record."""

    return _recent_jsonl(Path(path), limit)


def _series(
    events: list[dict[str, Any]], x_key: str, y_key: str
) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float]] = []
    for event in events:
        try:
            x = float(event[x_key])
            y = float(event[y_key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append((x, y))
    if not points:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    return (
        np.asarray([point[0] for point in points], dtype=np.float64),
        np.asarray([point[1] for point in points], dtype=np.float64),
    )


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    window = max(1, min(int(window), values.size))
    valid = np.isfinite(values)
    totals = np.cumsum(np.where(valid, values, 0.0))
    counts = np.cumsum(valid.astype(np.int64))
    totals = np.concatenate(([0.0], totals))
    counts = np.concatenate(([0], counts))
    starts = np.maximum(np.arange(values.size) + 1 - window, 0)
    sums = totals[np.arange(values.size) + 1] - totals[starts]
    samples = counts[np.arange(values.size) + 1] - counts[starts]
    return np.divide(
        sums,
        samples,
        out=np.full(values.size, np.nan, dtype=np.float64),
        where=samples > 0,
    )


def _downsample(x: np.ndarray, *ys: np.ndarray, limit: int = 1500):
    if x.size <= limit:
        return (x, *ys)
    indices = np.linspace(0, x.size - 1, limit, dtype=np.int64)
    return (x[indices], *(values[indices] for values in ys))


def _style_axis(axis, title: str, *, percent: bool = False) -> None:
    axis.set_title(title, loc="left", fontsize=10, fontweight="semibold")
    axis.grid(alpha=0.18, linewidth=0.7)
    axis.tick_params(labelsize=8)
    if percent:
        axis.set_ylim(0.0, 100.0)


def _create_training_figure(plt):
    plt.style.use("dark_background")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    figure = plt.figure(figsize=(19, 10.2))
    grid = figure.add_gridspec(
        3,
        4,
        height_ratios=(1.08, 1.0, 1.0),
        width_ratios=(1.05, 1.0, 1.05, 1.15),
        hspace=0.42,
        wspace=0.28,
    )
    axes = {
        "health": figure.add_subplot(grid[0, :3]),
        "status": figure.add_subplot(grid[0, 3]),
        "damage": figure.add_subplot(grid[1, 0]),
        "reward": figure.add_subplot(grid[1, 1]),
        "survival": figure.add_subplot(grid[1, 2]),
        "actions": figure.add_subplot(grid[1, 3]),
        "loss": figure.add_subplot(grid[2, 0]),
        "q": figure.add_subplot(grid[2, 1]),
        "runtime": figure.add_subplot(grid[2, 2]),
        "safeguards": figure.add_subplot(grid[2, 3]),
    }
    axes["runtime_rate"] = axes["runtime"].twinx()
    return figure, axes


def _current_episode_steps(actor_steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not actor_steps:
        return []
    episode_id = actor_steps[-1].get("episode_id")
    return [event for event in actor_steps if event.get("episode_id") == episode_id]


def _plot_episode_history(
    axes: dict[str, Any],
    episodes: list[dict[str, Any]],
    metric_episodes: list[dict[str, Any]],
) -> None:
    x = np.arange(1, len(episodes) + 1, dtype=np.float64)
    damages = np.asarray([event.get("damage_dealt", np.nan) for event in episodes], dtype=float)
    rewards = np.asarray([event.get("reward", np.nan) for event in episodes], dtype=float)
    steps = np.asarray([event.get("steps", np.nan) for event in episodes], dtype=float)
    wins = np.asarray([event.get("state") == "won" for event in episodes], dtype=bool)
    rolling_window = min(10, max(3, len(episodes) // 4)) if episodes else 1

    damage_axis = axes["damage"]
    damage_axis.bar(x, damages, color=np.where(wins, "#35d07f", "#4388cc"), alpha=0.72)
    if x.size:
        damage_axis.plot(x, _rolling_mean(damages, rolling_window), color="#ffd166", linewidth=2, label=f"{rolling_window}局均值")
        damage_axis.legend(fontsize=7, loc="upper left")
    damage_axis.set_xlabel("最近完整回合（旧 → 新）", fontsize=8)
    damage_axis.set_ylabel("伤害 %", fontsize=8)
    _style_axis(damage_axis, "Boss 伤害趋势", percent=True)

    reward_axis = axes["reward"]
    reward_axis.axhline(0.0, color="#8b9eb7", linewidth=0.8)
    reward_axis.bar(x, rewards, color=np.where(rewards >= 0.0, "#35d07f", "#ef6262"), alpha=0.72)
    if x.size:
        reward_axis.plot(x, _rolling_mean(rewards, rolling_window), color="#ffd166", linewidth=2, label=f"{rolling_window}局均值")
        by_episode = {event.get("episode_id"): event for event in metric_episodes}
        components = (
            ("Boss", "reward_boss_damage", "#35d07f"),
            ("自身", "reward_self_damage", "#ef6262"),
            ("时间", "reward_tick", "#58a6ff"),
            ("终局", "reward_terminal", "#ff9f43"),
        )
        for label, key, color in components:
            values = np.asarray(
                [by_episode.get(event.get("episode_id"), {}).get(key, np.nan) for event in episodes],
                dtype=float,
            )
            if np.isfinite(values).any():
                reward_axis.plot(x, values, color=color, linewidth=0.9, alpha=0.85, label=label)
        reward_axis.legend(fontsize=6.5, loc="upper left", ncols=2)
    reward_axis.set_xlabel("最近完整回合", fontsize=8)
    reward_axis.set_ylabel("总奖励", fontsize=8)
    _style_axis(reward_axis, "回合奖励趋势")

    survival_axis = axes["survival"]
    survival_axis.bar(x, steps, color=np.where(wins, "#35d07f", "#9a7bd1"), alpha=0.72)
    if x.size:
        survival_axis.plot(x, _rolling_mean(steps, rolling_window), color="#ffd166", linewidth=2, label=f"{rolling_window}局均值")
        survival_axis.legend(fontsize=7, loc="upper left")
    survival_axis.set_xlabel("最近完整回合", fontsize=8)
    survival_axis.set_ylabel("8Hz steps", fontsize=8)
    _style_axis(survival_axis, "生存时长")


def _plot_actions(axis, replay: dict[str, Any]) -> None:
    entries: list[tuple[str, float, str]] = []
    entries.extend((f"移动 · {name}", ratio, "#58a6ff") for name, ratio in replay.get("movement", []))
    entries.extend((f"战斗 · {name}", ratio, "#c77dff") for name, ratio in replay.get("combat", []))
    entries = list(reversed(entries))
    if entries:
        labels = [entry[0] for entry in entries]
        values = [entry[1] * 100.0 for entry in entries]
        colors = [entry[2] for entry in entries]
        axis.barh(labels, values, color=colors, alpha=0.82)
        for index, value in enumerate(values):
            axis.text(value + 0.6, index, f"{value:.1f}%", va="center", fontsize=7)
        axis.set_xlim(0.0, max(35.0, max(values) * 1.25))
    axis.set_xlabel("最近 5000 steps 占比", fontsize=8)
    _style_axis(axis, "动作分布")


def _plot_health(axis, actor_steps: list[dict[str, Any]]) -> None:
    current = _current_episode_steps(actor_steps)
    x, boss = _series(current, "step_id", "boss_health")
    _, player = _series(current, "step_id", "self_health")
    if x.size and player.size == x.size:
        x, boss, player = _downsample(x, boss, player)
        axis.plot(x, boss, color="#ef6262", linewidth=2.2, label="Boss HP")
        axis.plot(x, player, color="#35d07f", linewidth=2.0, label="自身 HP")
        axis.fill_between(x, boss, alpha=0.08, color="#ef6262")
        axis.legend(loc="upper right", ncols=2, fontsize=8)
    else:
        axis.text(0.5, 0.5, "等待当前回合指标…", ha="center", va="center", transform=axis.transAxes, color="#8b9eb7")
    axis.set_xlabel("当前回合 step（8Hz）", fontsize=8)
    axis.set_ylabel("HP %", fontsize=8)
    _style_axis(axis, "当前 / 最新回合血量曲线", percent=True)


def _plot_learner(axes: dict[str, Any], learner_steps: list[dict[str, Any]]) -> None:
    loss_axis = axes["loss"]
    x, td = _series(learner_steps, "learner_steps", "td_loss")
    _, demo = _series(learner_steps, "learner_steps", "demo_loss")
    _, total = _series(learner_steps, "learner_steps", "loss")
    if x.size and td.size == demo.size == total.size:
        x, td, demo, total = _downsample(x, td, demo, total)
        loss_axis.plot(x, _rolling_mean(total, 25), color="#ffd166", linewidth=1.5, label="总 loss")
        loss_axis.plot(x, _rolling_mean(td, 25), color="#58a6ff", linewidth=1.4, label="TD loss")
        loss_axis.plot(x, _rolling_mean(demo, 25), color="#c77dff", linewidth=1.4, label="示范 loss")
        loss_axis.legend(fontsize=7, ncols=2)
    loss_axis.set_xlabel("Learner update", fontsize=8)
    _style_axis(loss_axis, "Learner 损失（25步平滑）")

    q_axis = axes["q"]
    qx, mean_q = _series(learner_steps, "learner_steps", "mean_q")
    _, target = _series(learner_steps, "learner_steps", "mean_target")
    if qx.size and mean_q.size == target.size:
        qx, mean_q, target = _downsample(qx, mean_q, target)
        q_axis.plot(qx, _rolling_mean(mean_q, 25), color="#35d07f", linewidth=1.6, label="Q")
        q_axis.plot(qx, _rolling_mean(target, 25), color="#ff9f43", linewidth=1.6, label="Target")
        q_axis.legend(fontsize=7)
    q_axis.set_xlabel("Learner update", fontsize=8)
    _style_axis(q_axis, "Q 值与目标值（25步平滑）")


def _plot_runtime(axes: dict[str, Any], actor_steps: list[dict[str, Any]]) -> None:
    axis = axes["runtime"]
    rate_axis = axes["runtime_rate"]
    x, actor_latency = _series(actor_steps, "environment_steps", "actor_latency_ms")
    _, observation_latency = _series(actor_steps, "environment_steps", "observation_latency_ms")
    timestamps = np.asarray(
        [
            float(event.get("actor_timestamp", event.get("timestamp", np.nan)))
            for event in actor_steps
        ],
        dtype=np.float64,
    )
    deadline = np.asarray(
        [float(event.get("deadline_miss_rate", np.nan)) * 100.0 for event in actor_steps],
        dtype=np.float64,
    )
    fps = np.full(timestamps.size, np.nan, dtype=np.float64)
    if timestamps.size > 1:
        intervals = np.diff(timestamps)
        valid = (intervals > 0.0) & (intervals <= 0.5)
        fps[1:][valid] = 1.0 / intervals[valid]
    if x.size and actor_latency.size == observation_latency.size == timestamps.size:
        smooth_actor = _rolling_mean(actor_latency, 32)
        smooth_observation = _rolling_mean(observation_latency, 32)
        smooth_fps = _rolling_mean(fps, 32)
        smooth_deadline = _rolling_mean(deadline, 32)
        x, smooth_actor, smooth_observation, smooth_fps, smooth_deadline = _downsample(
            x, smooth_actor, smooth_observation, smooth_fps, smooth_deadline
        )
        axis.plot(x, smooth_actor, color="#58a6ff", linewidth=1.3, label="推理 ms")
        axis.plot(x, smooth_observation, color="#c77dff", linewidth=1.3, label="观测 ms")
        axis.axhline(15.0, color="#58a6ff", linestyle=":", linewidth=0.8)
        axis.axhline(20.0, color="#c77dff", linestyle=":", linewidth=0.8)
        rate_axis.plot(x, smooth_fps, color="#35d07f", linewidth=1.2, label="控制 FPS")
        rate_axis.plot(x, smooth_deadline, color="#ef6262", linewidth=1.1, label="miss %")
        lines = axis.get_lines()[:2] + rate_axis.get_lines()
        axis.legend(lines, [line.get_label() for line in lines], fontsize=7, ncols=2, loc="upper left")
    axis.set_xlabel("环境 step", fontsize=8)
    axis.set_ylabel("延迟 ms", fontsize=8)
    rate_axis.set_ylabel("FPS / miss %", fontsize=8)
    rate_axis.tick_params(labelsize=8)
    rate_axis.grid(False)
    _style_axis(axis, "控制性能（32步平滑）")


def _plot_safeguards(axis, actor_steps: list[dict[str, Any]]) -> None:
    x = np.asarray(
        [float(event.get("environment_steps", np.nan)) for event in actor_steps],
        dtype=np.float64,
    )
    definitions = (
        ("喝药开放", "potion_allowed", "#58a6ff"),
        ("变身开放", "skill4_allowed", "#ff9f43"),
        ("变身中", "transformation_active", "#c77dff"),
        ("策略被纠正", "policy_intervention", "#ef6262"),
    )
    for label, key, color in definitions:
        values = np.asarray(
            [float(bool(event.get(key, False))) * 100.0 for event in actor_steps],
            dtype=np.float64,
        )
        if x.size:
            plot_x, smooth = _downsample(x, _rolling_mean(values, 64))
            axis.plot(plot_x, smooth, color=color, linewidth=1.3, label=label)
    if x.size:
        axis.legend(fontsize=7, ncols=2, loc="upper left")
    axis.set_xlabel("环境 step", fontsize=8)
    axis.set_ylabel("64步占比 %", fontsize=8)
    _style_axis(axis, "动作掩码与安全干预", percent=True)


def _status_lines(snapshot: dict[str, Any]) -> list[str]:
    status = snapshot_status(snapshot)
    replay = snapshot.get("replay", {})
    current = replay.get("current", {})
    recent = snapshot.get("recent", {})
    actor = snapshot.get("actor", {})
    learner = snapshot.get("learner_summary", {})
    episode = snapshot.get("latest_episode_metric", {})
    lines = [f"● {status}", time_string(snapshot.get("timestamp", 0.0)), ""]
    if current:
        lines.extend(
            [
                f"回合 {current.get('episode_id', '-')}  ·  step {current.get('steps', '-')}",
                f"Boss {current.get('boss_health', math.nan):.1f}%  /  自身 {current.get('self_health', math.nan):.1f}%",
                f"当前奖励 {current.get('reward', math.nan):+.3f}",
            ]
        )
    if recent:
        count = int(recent.get("count", 0))
        lines.extend(
            [
                f"最近 {count} 局：{int(recent.get('wins', 0))} 胜  ·  均奖 {recent.get('reward_mean', math.nan):+.2f}",
                f"均伤 {recent.get('damage_mean', math.nan):.1f}%  /  最佳 {recent.get('damage_best', math.nan):.1f}%",
            ]
        )
    if actor:
        policy = snapshot.get("policy_summary", {})
        lines.extend(
            [
                f"控制 {policy.get('control_fps', math.nan):.2f} FPS  ·  miss {actor.get('deadline_miss_rate', math.nan):.1%}",
                f"推理 {actor.get('actor_latency_ms', math.nan):.1f}ms  /  观测 {actor.get('observation_latency_ms', math.nan):.1f}ms",
                f"感知置信度 {actor.get('detection_confidence', math.nan):.3f}",
            ]
        )
    if learner:
        lines.extend(
            [
                f"Learner {learner.get('updates_per_second', math.nan):.2f}/s  ·  TD {learner.get('td_loss', math.nan):.4f}",
                f"Q {learner.get('mean_q', math.nan):.3f}  /  target {learner.get('mean_target', math.nan):.3f}",
            ]
        )
    if episode and "reward_boss_damage" in episode:
        lines.extend(
            [
                "最近奖励分解",
                f"Boss {episode['reward_boss_damage']:+.2f} / 自身 {episode['reward_self_damage']:+.2f}",
                f"时间 {episode['reward_tick']:+.2f} / 终局 {episode['reward_terminal']:+.2f}",
            ]
        )
    return lines


def time_string(timestamp: float) -> str:
    import time

    if not timestamp:
        return "尚无指标"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def draw_training_dashboard(figure, axes: dict[str, Any], snapshot: dict[str, Any]) -> None:
    for name, axis in axes.items():
        axis.clear()
        if name != "status":
            axis.set_facecolor("#111820")

    actor_steps = list(snapshot.get("actor_steps", []))
    learner_steps = list(snapshot.get("learner_steps", []))
    episode_window = int(snapshot.get("episode_window", 50))
    episodes = list(snapshot.get("replay", {}).get("episodes", []))[-episode_window:]

    _plot_health(axes["health"], actor_steps)
    _plot_episode_history(axes, episodes, list(snapshot.get("metric_episodes", [])))
    _plot_actions(axes["actions"], snapshot.get("replay", {}))
    _plot_learner(axes, learner_steps)
    _plot_runtime(axes, actor_steps)
    _plot_safeguards(axes["safeguards"], actor_steps)

    status = snapshot_status(snapshot)
    status_axis = axes["status"]
    status_axis.axis("off")
    status_axis.set_facecolor("#0d141b")
    status_axis.text(
        0.04,
        0.98,
        "\n".join(_status_lines(snapshot)),
        va="top",
        ha="left",
        fontsize=8.5,
        linespacing=1.28,
        color="#dce6f0",
        transform=status_axis.transAxes,
    )
    status_axis.text(
        0.04,
        0.98,
        f"● {status}",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=_STATUS_COLORS[status],
        transform=status_axis.transAxes,
    )
    figure.suptitle(
        "黑神话：悟空 · R2D3 在线训练仪表盘",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )


def run_dashboard(
    metrics_directory: str | Path = "artifacts/metrics",
    refresh_seconds: float = 2.0,
    *,
    replay_directory: str | Path = "artifacts/replay/online",
    episode_window: int = 50,
    snapshot_path: str | Path | None = None,
) -> None:
    if refresh_seconds < 1.0:
        raise ValueError("refresh_seconds must be at least 1")
    if episode_window < 2:
        raise ValueError("episode_window must be at least 2")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as error:
        raise RuntimeError("install the dashboard extra: pip install -e '.[dashboard]'") from error

    figure, axes = _create_training_figure(plt)

    def redraw(_frame=None) -> None:
        snapshot = load_snapshot(
            metrics_directory,
            replay_directory,
            episode_window,
            include_series=True,
        )
        draw_training_dashboard(figure, axes, snapshot)
        figure.canvas.draw_idle()

    redraw()
    if snapshot_path is not None:
        output = Path(snapshot_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=150, bbox_inches="tight", facecolor=figure.get_facecolor())
        plt.close(figure)
        print(f"[dashboard] 图像已保存: {output.resolve()}", flush=True)
        return

    animation = FuncAnimation(
        figure,
        redraw,
        interval=max(int(refresh_seconds * 1000), 1000),
        cache_frame_data=False,
    )
    figure._wukong_animation = animation
    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager, "window") and hasattr(manager.window, "state"):
            manager.window.state("zoomed")
    except Exception:
        pass
    plt.show()


def run_profile_dashboard(directory: str | Path, refresh_seconds: float = 5.0) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as error:
        raise RuntimeError("install the dashboard extra: pip install -e '.[dashboard]'") from error
    root = Path(directory)
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    figure.suptitle("Recording performance — NOT game FPS (no in-game OSD)")

    def redraw(_frame=None):
        try:
            summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for axis in axes:
            axis.clear()
        stats = {
            name: value["p95"]
            for name, value in summary["stats"].items()
            if name.endswith(".wall_ms")
            and ".sleep." not in name
            and value["p95"] is not None
        }
        top = sorted(stats.items(), key=lambda item: item[1], reverse=True)[:12]
        axes[0].barh([item[0] for item in top], [item[1] for item in top])
        axes[0].invert_yaxis()
        axes[0].set_title("Stage p95 wall time (ms, rolling window)")
        axes[0].grid(axis="x", alpha=0.25)
        lines = [
            f"State: {summary['last_tick'].get('state', 'starting')}",
            f"Status: {summary['end_reason']}",
            f"Saved episodes: {summary['saved_episodes']}",
            f"Monitoring events dropped: {summary['dropped_events']}",
            "",
        ]
        for name, phase in summary["phases"].items():
            hz = phase.get("observed_loop_hz")
            invalid = (
                f"{phase['invalid_hp_rate']:.1%}"
                if phase["invalid_hp_rate"] is not None
                else "N/A"
            )
            lines += [
                f"{name}: loop Hz={hz:.2f}" if hz is not None else f"{name}: loop Hz=N/A",
                f"  deadline miss={phase['deadline_miss_rate']:.1%}; invalid HP={invalid}",
            ]
        resource = summary.get("latest_resource", {})
        for group in ("recorder", "monitor", "gpu"):
            values = resource.get(group) or {}
            fields = (
                ("utilization_percent", "memory_used_mb")
                if group == "gpu"
                else ("cpu_percent_one_core", "rss_mb")
            )
            lines += [
                f"{group}: "
                + ", ".join(f"{key}={values.get(key, 'N/A')}" for key in fields)
            ]
        lines += [
            "",
            "Game FPS: import PresentMon CSV separately.",
            "GPU utilization/memory is device-wide.",
        ]
        lines += [
            f"Unavailable {key}: {value}"
            for key, value in resource.get("unavailable", {}).items()
        ]
        axes[1].axis("off")
        axes[1].text(0, 1, "\n".join(lines), va="top", fontsize=9, wrap=True)
        figure.tight_layout()

    redraw()
    animation = FuncAnimation(
        figure,
        redraw,
        interval=max(1000, int(refresh_seconds * 1000)),
        cache_frame_data=False,
    )
    figure._wukong_animation = animation
    plt.show()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wukong RL graphical dashboard")
    parser.add_argument("--config", default="config/rl_pipeline.yaml")
    parser.add_argument("--metrics")
    parser.add_argument("--replay")
    parser.add_argument("--refresh", "--refresh-seconds", dest="refresh", type=float, default=2.0)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--snapshot", help="render once to a PNG instead of opening a live window")
    parser.add_argument("--profile", help="performance run directory; reads only bounded summary.json")
    args = parser.parse_args(argv)
    if args.profile:
        run_profile_dashboard(args.profile, args.refresh)
    else:
        from .config import load_config, online_replay_directory

        config = load_config(args.config)
        run_dashboard(
            args.metrics or config.training.metrics_directory,
            args.refresh,
            replay_directory=args.replay or online_replay_directory(config),
            episode_window=args.window,
            snapshot_path=args.snapshot,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
