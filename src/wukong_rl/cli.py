from __future__ import annotations

import argparse
import json

from .agent import R2D3Agent
from .checkpoint import save_checkpoint
from .config import load_config, write_migrated_config
from .data import TrajectoryDataset
from .evaluation import evaluate_live
from .metrics import JsonlMetricWriter
from .pretrain import BehaviorCloningTrainer
from .recording import record_demonstrations
from .runtime import run_training
from .tools import benchmark, calibrate
from .types import ActionToken, HUD_KEYS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wukong screen-control RL pipeline")
    parser.add_argument("--config", default="config/rl_pipeline.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("--output", default="artifacts/calibration/yinhu.png")
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--boss", default="yinhu")
    record_parser.add_argument("--output", default=None)
    pretrain_parser = subparsers.add_parser("pretrain")
    pretrain_parser.add_argument("--dataset", required=True)
    pretrain_parser.add_argument("--epochs", type=int, default=10)
    pretrain_parser.add_argument("--steps-per-epoch", type=int, default=100)
    pretrain_parser.add_argument("--validation-steps", type=int, default=20)
    pretrain_parser.add_argument("--checkpoint", default="artifacts/checkpoints/bc-pretrained.pt")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--boss", default="yinhu")
    train_parser.add_argument("--dataset")
    train_parser.add_argument("--checkpoint")
    eval_parser = subparsers.add_parser("eval")
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
    config = load_config(args.config)
    if hasattr(args, "boss"):
        config.environment.boss_id = args.boss
    if args.command == "calibrate":
        print(calibrate(config, args.output))
    elif args.command == "record":
        record_demonstrations(config, args.boss, args.output or config.training.dataset_directory)
    elif args.command == "pretrain":
        dataset = TrajectoryDataset(args.dataset, boss_id=config.environment.boss_id)
        training_paths, validation_paths = dataset.split()
        agent = R2D3Agent(len(HUD_KEYS), ActionToken.size(), config.model)
        trainer = BehaviorCloningTrainer(agent, sequence_length=config.model.unroll)
        writer = JsonlMetricWriter(config.training.metrics_directory, "pretrain")
        best_loss = float("inf")
        for epoch in range(1, args.epochs + 1):
            train_metrics = trainer.run_epoch(
                training_paths,
                batch_size=config.model.batch_size,
                steps=args.steps_per_epoch,
                train=True,
            )
            validation_metrics = trainer.run_epoch(
                validation_paths,
                batch_size=config.model.batch_size,
                steps=args.validation_steps,
                train=False,
            )
            writer.write(
                "epoch",
                epoch=epoch,
                train_loss=train_metrics.loss,
                train_accuracy=train_metrics.accuracy,
                validation_loss=validation_metrics.loss,
                validation_accuracy=validation_metrics.accuracy,
                class_recall=validation_metrics.class_recall,
                confusion=validation_metrics.confusion,
            )
            print(
                f"epoch={epoch} train_loss={train_metrics.loss:.4f} "
                f"val_loss={validation_metrics.loss:.4f} "
                f"val_accuracy={validation_metrics.accuracy:.3f}"
            )
            if validation_metrics.loss < best_loss:
                best_loss = validation_metrics.loss
                agent.sync_target()
                save_checkpoint(
                    agent,
                    args.checkpoint,
                    config.fingerprint(),
                    {"stage": "behavior_cloning", "epoch": epoch},
                    data_version=dataset.version,
                )
    elif args.command == "train":
        dataset_path = args.dataset or config.training.dataset_directory
        TrajectoryDataset(dataset_path, boss_id=config.environment.boss_id)
        run_training(args.config, dataset_path, args.checkpoint, args.boss)
    elif args.command == "eval":
        summary = evaluate_live(config, args.checkpoint, args.episodes, args.exploration)
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0 if summary["passed"] else 2
    elif args.command == "benchmark":
        print(json.dumps(benchmark(config, args.iterations, args.live_capture), indent=2))
    return 0
