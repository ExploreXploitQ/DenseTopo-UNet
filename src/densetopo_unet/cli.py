"""English command-line interface for local DenseTopo-UNet workflows."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from densetopo_unet.checkpoint import load_checkpoint
from densetopo_unet.config import load_experiment_config
from densetopo_unet.engine import train
from densetopo_unet.inference import InferenceRequest, restore_file
from densetopo_unet.manifest import Split, load_manifest
from densetopo_unet.metrics import evaluate_restored


def _path(value: str) -> Path:
    return Path(value)


def _add_config_manifest(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=_path, help="Experiment YAML file.")
    parser.add_argument("--manifest", required=True, type=_path, help="Data manifest YAML file.")


def build_parser() -> argparse.ArgumentParser:
    """Construct the public argument parser without executing a command."""

    parser = argparse.ArgumentParser(
        prog="densetopo",
        description=(
            "Train and apply a gated 3D U-Net for topology restoration of "
            "lossy-decompressed scalar fields."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    validate_parser = commands.add_parser(
        "validate-manifest",
        help="Validate configuration, volume files, and topology-label metadata.",
    )
    _add_config_manifest(validate_parser)

    train_parser = commands.add_parser(
        "train",
        help="Train a new model or resume a compatible local run.",
    )
    _add_config_manifest(train_parser)
    train_parser.add_argument("--output", required=True, type=_path, help="New run directory.")
    train_parser.add_argument(
        "--resume",
        type=_path,
        help="Optional checkpoint from the same configuration and manifest.",
    )

    infer_parser = commands.add_parser(
        "infer",
        help="Restore one lossy-decompressed float32 volume.",
    )
    infer_parser.add_argument("--checkpoint", required=True, type=_path)
    infer_parser.add_argument(
        "--input",
        required=True,
        type=_path,
        help="Headerless float32 lossy-decompressed volume.",
    )
    infer_parser.add_argument(
        "--shape",
        required=True,
        nargs=3,
        type=int,
        metavar=("D", "H", "W"),
        help="Volume dimensions in [D, H, W] order.",
    )
    infer_parser.add_argument("--output", required=True, type=_path)
    infer_parser.add_argument(
        "--byte-order",
        choices=("little", "big"),
        default="little",
        help="Input and output float32 byte order (default: little).",
    )
    infer_parser.add_argument("--batch-size", type=int, default=2)
    infer_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="Evaluate restored volumes and optional external FC summaries.",
    )
    _add_config_manifest(evaluate_parser)
    evaluate_parser.add_argument("--restored-root", required=True, type=_path)
    evaluate_parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluate_parser.add_argument("--output", required=True, type=_path)
    evaluate_parser.add_argument("--baseline-topology-dir", type=_path)
    evaluate_parser.add_argument("--restored-topology-dir", type=_path)

    inspect_parser = commands.add_parser(
        "inspect-checkpoint",
        help="Print checkpoint metadata without constructing a GPU model.",
    )
    inspect_parser.add_argument("--checkpoint", required=True, type=_path)
    return parser


def _load_inputs(arguments: argparse.Namespace) -> tuple[Any, Any]:
    config = load_experiment_config(arguments.config)
    manifest = load_manifest(arguments.manifest, config.volume)
    return config, manifest


def _validate(arguments: argparse.Namespace) -> dict[str, Any]:
    config, manifest = _load_inputs(arguments)
    del config
    split_counts = {
        split: len(manifest.by_split(split)) for split in ("test", "train", "validation")
    }
    return {
        "valid": True,
        "experiment": manifest.experiment,
        "manifest": str(manifest.path),
        "sample_count": len(manifest.samples),
        "splits": split_counts,
    }


def _train(arguments: argparse.Namespace) -> dict[str, Any]:
    config, manifest = _load_inputs(arguments)
    return train(config, manifest, arguments.output, arguments.resume).to_dict()


def _infer(arguments: argparse.Namespace) -> dict[str, Any]:
    request = InferenceRequest(
        checkpoint=arguments.checkpoint,
        input_path=arguments.input,
        output_path=arguments.output,
        shape=tuple(arguments.shape),
        byte_order=arguments.byte_order,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )
    return restore_file(request).to_dict()


def _evaluate(arguments: argparse.Namespace) -> dict[str, Any]:
    config, manifest = _load_inputs(arguments)
    summary = evaluate_restored(
        config=config,
        manifest=manifest,
        restored_root=arguments.restored_root,
        split=cast(Split, arguments.split),
        output_dir=arguments.output,
        baseline_topology_dir=arguments.baseline_topology_dir,
        restored_topology_dir=arguments.restored_topology_dir,
    )
    return summary.to_dict()


def _inspect(arguments: argparse.Namespace) -> dict[str, Any]:
    state = load_checkpoint(arguments.checkpoint)
    return {
        "schema_version": state.schema_version,
        "package_version": state.package_version,
        "configuration": state.config,
        "manifest_sha256": state.manifest_sha256,
        "epoch": state.epoch,
        "best_score": state.best_score,
        "bad_epochs": state.bad_epochs,
        "history_rows": len(state.history),
        "model_tensor_count": len(state.model_state),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one command and return a process-compatible status code."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    handlers = {
        "validate-manifest": _validate,
        "train": _train,
        "infer": _infer,
        "evaluate": _evaluate,
        "inspect-checkpoint": _inspect,
    }
    try:
        result = handlers[arguments.command](arguments)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
