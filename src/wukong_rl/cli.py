from __future__ import annotations

import argparse
import json
import math
import time

from .config import load_config, write_migrated_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wukong screen-control RL pipeline")
    parser.add_argument("--config", default="config/rl_pipeline.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--output", default="artifacts/calibration/yinhu.png")
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--boss", default="yinhu")
    record_parser.add_argument("--output", default=None)
    record_parser.add_argument("--no-profile", action="store_true")
    record_parser.add_argument("--profile-dir")
    record_parser.add_argument("--seconds", type=float, help="optional recording duration")
    record_parser.add_argument("--start-paused", action="store_true", help="wait for F8 before recording")
    record_parser.add_argument("--immediate", action="store_true", help=argparse.SUPPRESS)
    diagnose_parser = subparsers.add_parser("diagnose", help="read-only performance A/B probe (no input injection)")
    diagnose_parser.add_argument("--mode", choices=("baseline", "capture", "observe", "offline"), default="observe")
    diagnose_parser.add_argument("--seconds", type=float, default=30.0)
    diagnose_parser.add_argument("--frame", help="saved BGR image for offline mode")
    diagnose_parser.add_argument("--profile-dir")
    telemetry_parser = subparsers.add_parser(
        "telemetry-probe",
        help="read-only named-pipe telemetry probe (no capture or input injection)",
    )
    telemetry_parser.add_argument("--seconds", type=float, default=15.0)
    report_parser = subparsers.add_parser("profile-report")
    report_parser.add_argument("--run", required=True)
    report_parser.add_argument("--compare")
    report_parser.add_argument("--presentmon-csv")
    report_parser.add_argument("--game-process", default="b1-Win64-Shipping.exe")
    pretrain_parser = subparsers.add_parser(
        "pretrain", help="supervised behavior-cloning training from demonstrations"
    )
    pretrain_parser.add_argument("--dataset", required=True)
    pretrain_parser.add_argument("--epochs", type=int, default=10)
    pretrain_parser.add_argument("--steps-per-epoch", type=int, default=100)
    pretrain_parser.add_argument("--validation-steps", type=int, default=20)
    pretrain_parser.add_argument("--checkpoint", default="artifacts/checkpoints/bc-pretrained.pt")
    repair_parser = subparsers.add_parser(
        "repair-dataset",
        help="non-destructively restore legacy Q potion labels from raw input",
    )
    repair_parser.add_argument("--dataset", required=True)
    repair_parser.add_argument("--output", required=True)
    repair_parser.add_argument("--boss", default="yinhu")
    train_parser = subparsers.add_parser(
        "train", help="online reinforcement learning with live game control"
    )
    train_parser.add_argument("--boss", default="yinhu")
    train_parser.add_argument("--dataset")
    train_parser.add_argument("--checkpoint")
    train_parser.add_argument(
        "--max-environment-steps",
        type=int,
        help="optional bounded live-training run for verification",
    )
    train_parser.add_argument(
        "--allow-unready-checkpoint",
        action="store_true",
        help="bypass the offline behavior-cloning release gate",
    )
    monitor_parser = subparsers.add_parser(
        "monitor", help="read-only live convergence and training-health dashboard"
    )
    monitor_parser.add_argument("--metrics")
    monitor_parser.add_argument("--replay")
    monitor_parser.add_argument("--refresh", type=float, default=5.0)
    monitor_parser.add_argument("--window", type=int, default=10)
    monitor_parser.add_argument("--once", action="store_true")
    eval_parser = subparsers.add_parser(
        "eval", help="frozen-policy evaluation only; never updates model weights"
    )
    eval_parser.add_argument("--checkpoint", default="artifacts/checkpoints/latest.pt")
    eval_parser.add_argument("--episodes", type=int, default=20)
    eval_parser.add_argument("--exploration", type=float, default=0.0)
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--iterations", type=int, default=100)
    benchmark_parser.add_argument("--live-capture", action="store_true")
    migrate_parser = subparsers.add_parser("migrate-config")
    migrate_parser.add_argument("--output", default="config/rl_pipeline.migrated.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "migrate-config":
        print(write_migrated_config(args.output))
        return 0
    if args.command == "profile-report":
        from .diagnostics import write_analysis

        print(write_analysis(args.run, args.compare, args.presentmon_csv, args.game_process))
        return 0
    config = load_config(args.config)
    if hasattr(args, "boss"):
        config.environment.boss_id = args.boss
    if args.command == "calibrate":
        from .tools import calibrate

        print(calibrate(config, args.output))
    elif args.command == "record":
        from .recording import record_demonstrations

        record_demonstrations(
            config, args.boss, args.output or config.training.dataset_directory,
            profile_enabled=False if args.no_profile else None,
            profile_directory=args.profile_dir, duration_seconds=args.seconds,
            start_paused=args.start_paused and not args.immediate,
        )
    elif args.command == "diagnose":
        from .diagnostics import diagnose

        print(diagnose(config, args.mode, args.seconds, args.frame, args.profile_dir))
    elif args.command == "telemetry-probe":
        from .telemetry import NamedPipeTelemetryClient, snapshot_summary

        if args.seconds <= 0:
            raise ValueError("--seconds must be positive")
        if config.telemetry.mode == "off":
            raise RuntimeError("telemetry is disabled; set telemetry.mode to prefer or required")
        client = NamedPipeTelemetryClient(config.telemetry)
        client.start()
        deadline = time.monotonic() + args.seconds
        last_sequence = None
        last_status_print = 0.0
        print(f"Telemetry probe started: \\\\.\\pipe\\{config.telemetry.pipe_name}")
        try:
            while time.monotonic() < deadline:
                now = time.monotonic()
                snapshot = client.latest(now)
                if (
                    snapshot is not None
                    and snapshot.sequence != last_sequence
                    and now - last_status_print >= 0.5
                ):
                    print(json.dumps(snapshot_summary(snapshot), ensure_ascii=False))
                    last_sequence = snapshot.sequence
                    last_status_print = now
                elif now - last_status_print >= 1.0:
                    status = client.status(now)
                    print(
                        json.dumps(
                            {
                                "connected": status.connected,
                                "fresh": status.fresh,
                                "age_ms": status.age_ms,
                                "valid_packets": status.valid_packets,
                                "invalid_packets": status.invalid_packets,
                                "last_error": status.last_error,
                            },
                            ensure_ascii=False,
                        )
                    )
                    last_status_print = now
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            status = client.status()
            client.close()
        return 0 if status.valid_packets else 2
    elif args.command == "pretrain":
        from pathlib import Path

        import torch

        from .agent import R2D3Agent
        from .checkpoint import save_checkpoint
        from .data import TrajectoryDataset
        from .metrics import JsonlMetricWriter
        from .pretrain import BehaviorCloningTrainer, assess_bc_release, core_balanced_score
        from .types import ACTION_MASK_SIZE, HUD_KEYS

        torch.manual_seed(config.training.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.random_seed)
        dataset = TrajectoryDataset(args.dataset, boss_id=config.environment.boss_id)
        training_paths, validation_paths = dataset.split()
        print(
            f"dataset={dataset.version} train_episodes={len(training_paths)} "
            f"validation_episodes={len(validation_paths)}",
            flush=True,
        )
        agent = R2D3Agent(len(HUD_KEYS), ACTION_MASK_SIZE, config.model)
        trainer = BehaviorCloningTrainer(
            agent,
            sequence_length=config.model.unroll,
            burn_in=config.model.burn_in,
            seed=config.training.random_seed,
        )
        class_weights = trainer.estimate_class_weights(training_paths)
        writer = JsonlMetricWriter(config.training.metrics_directory, "pretrain")
        best_loss = float("inf")
        best_rank = (-1, float("-inf"))
        best_release_ready = False
        best_release_reasons: list[str] = []
        checkpoint = Path(args.checkpoint)
        best_loss_checkpoint = checkpoint.with_name(
            f"{checkpoint.stem}-best-loss{checkpoint.suffix}"
        )
        last_checkpoint = checkpoint.with_name(f"{checkpoint.stem}-last{checkpoint.suffix}")
        for epoch in range(1, args.epochs + 1):
            # Reach full self-feedback halfway through training, leaving
            # several epochs to learn recovery from its own action history.
            feedback_ramp_epochs = max(2, int(math.ceil(args.epochs * 0.5)))
            model_feedback_probability = min(
                1.0, (epoch - 1) / max(feedback_ramp_epochs - 1, 1)
            )
            train_metrics = trainer.run_epoch(
                training_paths,
                batch_size=config.model.batch_size,
                steps=args.steps_per_epoch,
                train=True,
                class_weights=class_weights,
                model_feedback_probability=model_feedback_probability,
            )
            validation_metrics = trainer.run_epoch(
                validation_paths,
                batch_size=config.model.batch_size,
                steps=args.validation_steps,
                train=False,
                class_weights=class_weights,
                model_feedback_probability=1.0,
            )
            closed_loop_metrics = trainer.evaluate_closed_loop(validation_paths)
            release_ready, release_reasons = assess_bc_release(
                closed_loop_metrics, validation_metrics
            )
            selection_score = core_balanced_score(validation_metrics, closed_loop_metrics)
            writer.write(
                "epoch",
                epoch=epoch,
                train_loss=train_metrics.loss,
                train_accuracy=train_metrics.accuracy,
                validation_loss=validation_metrics.loss,
                validation_accuracy=validation_metrics.joint_accuracy,
                validation_movement_accuracy=validation_metrics.movement_accuracy,
                validation_combat_accuracy=validation_metrics.combat_accuracy,
                core_balanced_score=selection_score,
                movement_recall=validation_metrics.movement_recall,
                combat_recall=validation_metrics.combat_recall,
                movement_confusion=validation_metrics.movement_confusion,
                combat_confusion=validation_metrics.combat_confusion,
                closed_loop_joint_accuracy=closed_loop_metrics.joint_accuracy,
                closed_loop_movement_accuracy=closed_loop_metrics.movement_accuracy,
                closed_loop_combat_accuracy=closed_loop_metrics.combat_accuracy,
                static_escape_rate=closed_loop_metrics.static_escape_rate,
                mean_first_action_step=closed_loop_metrics.mean_first_action_step,
                max_idle_run=closed_loop_metrics.max_idle_run,
                closed_loop_movement_active_rate=closed_loop_metrics.movement_active_rate,
                closed_loop_combat_active_rate=closed_loop_metrics.combat_active_rate,
                closed_loop_movement_switch_rate=closed_loop_metrics.movement_switch_rate,
                closed_loop_combat_switch_rate=closed_loop_metrics.combat_switch_rate,
                closed_loop_movement_counts=closed_loop_metrics.movement_counts,
                closed_loop_combat_counts=closed_loop_metrics.combat_counts,
                release_ready=release_ready,
                release_reasons=release_reasons,
                model_feedback_probability=model_feedback_probability,
            )
            print(
                f"epoch={epoch} train_loss={train_metrics.loss:.4f} "
                f"val_loss={validation_metrics.loss:.4f} "
                f"val_joint={validation_metrics.joint_accuracy:.3f} "
                f"val_move={validation_metrics.movement_accuracy:.3f} "
                f"val_combat={validation_metrics.combat_accuracy:.3f} "
                f"core_balanced_score={selection_score:.3f} "
                f"static_escape={closed_loop_metrics.static_escape_rate:.2f} "
                f"max_idle={closed_loop_metrics.max_idle_run} "
                f"release_ready={release_ready} "
                f"model_feedback={model_feedback_probability:.2f}"
            )
            agent.sync_target()
            common_extra = {
                "stage": "behavior_cloning",
                "epoch": epoch,
                "validation_loss": validation_metrics.loss,
                "validation_accuracy": validation_metrics.joint_accuracy,
                "validation_movement_accuracy": validation_metrics.movement_accuracy,
                "validation_combat_accuracy": validation_metrics.combat_accuracy,
                "core_balanced_score": selection_score,
                "burn_in": config.model.burn_in,
                "unroll": config.model.unroll,
                "model_feedback_probability": model_feedback_probability,
                "validation_model_feedback_probability": 1.0,
                "closed_loop_joint_accuracy": closed_loop_metrics.joint_accuracy,
                "static_escape_rate": closed_loop_metrics.static_escape_rate,
                "max_idle_run": closed_loop_metrics.max_idle_run,
                "closed_loop_movement_active_rate": closed_loop_metrics.movement_active_rate,
                "closed_loop_combat_active_rate": closed_loop_metrics.combat_active_rate,
                "closed_loop_movement_switch_rate": closed_loop_metrics.movement_switch_rate,
                "closed_loop_combat_switch_rate": closed_loop_metrics.combat_switch_rate,
                "closed_loop_movement_counts": closed_loop_metrics.movement_counts,
                "closed_loop_combat_counts": closed_loop_metrics.combat_counts,
                "release_ready": release_ready,
                "release_reasons": release_reasons,
                "action_space": "branched-v1",
            }
            save_checkpoint(
                agent,
                last_checkpoint,
                config.fingerprint(),
                {**common_extra, "selection": "last"},
                data_version=dataset.version,
            )
            if validation_metrics.loss < best_loss:
                best_loss = validation_metrics.loss
                save_checkpoint(
                    agent,
                    best_loss_checkpoint,
                    config.fingerprint(),
                    {**common_extra, "selection": "best_validation_loss"},
                    data_version=dataset.version,
                )
            selection_rank = (int(release_ready), selection_score)
            if selection_rank > best_rank:
                best_rank = selection_rank
                best_release_ready = release_ready
                best_release_reasons = release_reasons
                save_checkpoint(
                    agent,
                    checkpoint,
                    config.fingerprint(),
                    {**common_extra, "selection": "core_balanced"},
                    data_version=dataset.version,
                )
        print(
            f"[pretrain] selected_checkpoint={checkpoint} "
            f"release_ready={best_release_ready} reasons={best_release_reasons}",
            flush=True,
        )
        if not best_release_ready:
            return 2
    elif args.command == "repair-dataset":
        from dataclasses import asdict

        from .data import repair_legacy_potion_inputs

        result = repair_legacy_potion_inputs(
            args.dataset, args.output, boss_id=args.boss
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    elif args.command == "train":
        from pathlib import Path

        from .checkpoint import checkpoint_metadata
        from .data import TrajectoryDataset
        from .runtime import run_training

        dataset_path = args.dataset or config.training.dataset_directory
        TrajectoryDataset(dataset_path, boss_id=config.environment.boss_id)
        if not args.checkpoint and not args.allow_unready_checkpoint:
            raise RuntimeError(
                "--checkpoint is required; pretrain and pass the offline release gate first"
            )
        if args.checkpoint and Path(args.checkpoint).exists():
            metadata = checkpoint_metadata(args.checkpoint)
            extra = metadata.get("extra", {})
            if (
                extra.get("stage") == "behavior_cloning"
                and not extra.get("release_ready", False)
                and not args.allow_unready_checkpoint
            ):
                raise RuntimeError(
                    "behavior-cloning checkpoint did not pass the offline release gate: "
                    f"{extra.get('release_reasons', ['release metadata missing'])}"
                )
        run_training(
            args.config,
            dataset_path,
            args.checkpoint,
            args.boss,
            max_environment_steps=args.max_environment_steps,
        )
    elif args.command == "monitor":
        from .config import online_replay_directory
        from .monitor import run_monitor

        run_monitor(
            args.metrics or config.training.metrics_directory,
            args.replay or online_replay_directory(config),
            refresh_seconds=args.refresh,
            episode_window=args.window,
            once=args.once,
        )
    elif args.command == "eval":
        from .evaluation import evaluate_live

        summary = evaluate_live(config, args.checkpoint, args.episodes, args.exploration)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary["passed"] else 2
    elif args.command == "benchmark":
        from .tools import benchmark

        print(json.dumps(benchmark(config, args.iterations, args.live_capture), indent=2))
    return 0
