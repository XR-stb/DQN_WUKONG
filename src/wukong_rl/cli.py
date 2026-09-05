from __future__ import annotations

import argparse
import json

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
    train_parser = subparsers.add_parser(
        "train", help="online reinforcement learning with live game control"
    )
    train_parser.add_argument("--boss", default="yinhu")
    train_parser.add_argument("--dataset")
    train_parser.add_argument("--checkpoint")
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
    elif args.command == "pretrain":
        from pathlib import Path

        import torch

        from .agent import R2D3Agent
        from .checkpoint import save_checkpoint
        from .data import TrajectoryDataset
        from .metrics import JsonlMetricWriter
        from .pretrain import BehaviorCloningTrainer, core_balanced_score
        from .types import ActionToken, HUD_KEYS

        torch.manual_seed(config.training.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.random_seed)
        dataset = TrajectoryDataset(args.dataset, boss_id=config.environment.boss_id)
        training_paths, validation_paths = dataset.split()
        agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config.model)
        trainer = BehaviorCloningTrainer(
            agent,
            sequence_length=config.model.unroll,
            burn_in=config.model.burn_in,
            seed=config.training.random_seed,
        )
        class_weights = trainer.estimate_class_weights(training_paths)
        writer = JsonlMetricWriter(config.training.metrics_directory, "pretrain")
        best_loss = float("inf")
        best_score = float("-inf")
        checkpoint = Path(args.checkpoint)
        best_loss_checkpoint = checkpoint.with_name(
            f"{checkpoint.stem}-best-loss{checkpoint.suffix}"
        )
        last_checkpoint = checkpoint.with_name(f"{checkpoint.stem}-last{checkpoint.suffix}")
        for epoch in range(1, args.epochs + 1):
            train_metrics = trainer.run_epoch(
                training_paths,
                batch_size=config.model.batch_size,
                steps=args.steps_per_epoch,
                train=True,
                class_weights=class_weights,
            )
            validation_metrics = trainer.run_epoch(
                validation_paths,
                batch_size=config.model.batch_size,
                steps=args.validation_steps,
                train=False,
                class_weights=class_weights,
            )
            selection_score = core_balanced_score(validation_metrics)
            writer.write(
                "epoch",
                epoch=epoch,
                train_loss=train_metrics.loss,
                train_accuracy=train_metrics.accuracy,
                validation_loss=validation_metrics.loss,
                validation_accuracy=validation_metrics.accuracy,
                core_balanced_score=selection_score,
                class_recall=validation_metrics.class_recall,
                confusion=validation_metrics.confusion,
            )
            print(
                f"epoch={epoch} train_loss={train_metrics.loss:.4f} "
                f"val_loss={validation_metrics.loss:.4f} "
                f"val_accuracy={validation_metrics.accuracy:.3f} "
                f"core_balanced_score={selection_score:.3f}"
            )
            agent.sync_target()
            common_extra = {
                "stage": "behavior_cloning",
                "epoch": epoch,
                "validation_loss": validation_metrics.loss,
                "validation_accuracy": validation_metrics.accuracy,
                "core_balanced_score": selection_score,
                "burn_in": config.model.burn_in,
                "unroll": config.model.unroll,
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
            if selection_score > best_score:
                best_score = selection_score
                save_checkpoint(
                    agent,
                    checkpoint,
                    config.fingerprint(),
                    {**common_extra, "selection": "core_balanced"},
                    data_version=dataset.version,
                )
    elif args.command == "train":
        from .data import TrajectoryDataset
        from .runtime import run_training

        dataset_path = args.dataset or config.training.dataset_directory
        TrajectoryDataset(dataset_path, boss_id=config.environment.boss_id)
        run_training(args.config, dataset_path, args.checkpoint, args.boss)
    elif args.command == "eval":
        from .evaluation import evaluate_live

        summary = evaluate_live(config, args.checkpoint, args.episodes, args.exploration)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary["passed"] else 2
    elif args.command == "benchmark":
        from .tools import benchmark

        print(json.dumps(benchmark(config, args.iterations, args.live_capture), indent=2))
    return 0
