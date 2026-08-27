"""Training and validation engine for DenseTopo-UNet."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, cast

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from densetopo_unet import __version__
from densetopo_unet.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointState,
    load_checkpoint,
    manifest_fingerprint,
    save_checkpoint_atomic,
)
from densetopo_unet.config import ExperimentConfig
from densetopo_unet.data import TopologyPatchDataset
from densetopo_unet.losses import LossBreakdown, LossWeights, compute_losses
from densetopo_unet.manifest import DataManifest
from densetopo_unet.model import DenseTopoUNet3D
from densetopo_unet.reproducibility import (
    capture_rng_state,
    cuda_is_usable,
    environment_report,
    restore_rng_state,
    seed_everything,
)


@dataclass(frozen=True)
class TrainingSummary:
    """Compact record returned by a completed training run."""

    stopped_epoch: int
    best_epoch: int
    best_score: float
    output_directory: Path

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["output_directory"] = str(self.output_directory)
        return result


def topology_warmup(epoch: int, start: int, end: int) -> float:
    """Return the piecewise-linear topology objective multiplier."""

    if epoch < 0 or start < 0 or end < start:
        raise ValueError("epoch and warm-up bounds must satisfy 0 <= start <= end")
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return 1.0
    if end == start:
        return 1.0
    return float(epoch - start) / float(end - start)


def select_device(name: str) -> torch.device:
    """Resolve an explicit or automatic training device."""

    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not cuda_is_usable():
            raise RuntimeError("CUDA was requested but no usable CUDA device is visible")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if cuda_is_usable() else "cpu")
    raise ValueError(f"unsupported device: {name}")


def _tensor(batch: Mapping[str, object], key: str, device: torch.device) -> torch.Tensor:
    value = batch[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"batch field {key} must be a tensor")
    return value.to(device, non_blocking=device.type == "cuda")


def run_epoch(
    model: DenseTopoUNet3D,
    loader: DataLoader[dict[str, torch.Tensor | str]],
    device: torch.device,
    topology_lambda: float,
    loss_weights: LossWeights,
    xi: float,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Run one training or validation epoch and average named losses."""

    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    amp_enabled = bool(scaler is not None and scaler.is_enabled())

    for raw_batch in loader:
        batch = cast(Mapping[str, object], raw_batch)
        normalized = _tensor(batch, "input", device)
        decompressed = _tensor(batch, "decompressed", device)
        target = _tensor(batch, "target", device)
        topo_weight = _tensor(batch, "topo_weight", device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = model(normalized, decompressed, xi=xi)
                losses = compute_losses(
                    output,
                    target,
                    decompressed,
                    topo_weight,
                    topology_lambda,
                    loss_weights,
                    xi,
                )
            if optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(losses.total).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses.total.backward()
                    optimizer.step()

        for name, value in losses.as_dict().items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().float().cpu())
        batches += 1

    if batches == 0:
        raise RuntimeError("data loader produced no batches")
    return {name: total / batches for name, total in totals.items()}


def _prepare_output(output: Path, resume: Path | None) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()) and resume is None:
        raise ValueError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)


def _loader(
    dataset: TopologyPatchDataset,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> DataLoader[dict[str, torch.Tensor | str]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _checkpoint_state(
    config: ExperimentConfig,
    manifest_hash: str,
    model: DenseTopoUNet3D,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_score: float,
    bad_epochs: int,
    history: list[dict[str, float]],
) -> CheckpointState:
    return CheckpointState(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        package_version=__version__,
        config=config.to_dict(),
        manifest_sha256=manifest_hash,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
        epoch=epoch,
        best_score=best_score,
        bad_epochs=bad_epochs,
        history=history,
        rng_state=capture_rng_state(),
    )


def _write_history(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(
    config: ExperimentConfig,
    manifest: DataManifest,
    output_dir: Path,
    resume: Path | None = None,
) -> TrainingSummary:
    """Train a model and write a complete reproducible local run directory."""

    output = output_dir.resolve()
    _prepare_output(output, resume)
    seed_everything(config.training.seed)
    device = select_device(config.training.device)
    manifest_hash = manifest_fingerprint(manifest.path)

    train_dataset = TopologyPatchDataset(
        manifest,
        config,
        "train",
        config.training.samples_per_epoch,
        config.training.seed,
        augment=True,
    )
    validation_dataset = TopologyPatchDataset(
        manifest,
        config,
        "validation",
        config.training.validation_samples,
        config.training.seed + 1_000_000,
        augment=False,
    )
    train_loader = _loader(
        train_dataset,
        config.training.batch_size,
        config.training.num_workers,
        device,
    )
    validation_loader = _loader(
        validation_dataset,
        config.training.validation_batch_size,
        config.training.num_workers,
        device,
    )

    model = DenseTopoUNet3D(
        base_channels=config.model.base_channels,
        correction_scale=config.model.correction_scale,
        nonnegative=config.volume.value_domain == "nonnegative",
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    use_amp = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_weights = LossWeights()

    start_epoch = 1
    best_score = math.inf
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float]] = []
    if resume is not None:
        state = load_checkpoint(resume, expected_config=config)
        if state.manifest_sha256 != manifest_hash:
            raise CheckpointError("resume checkpoint manifest fingerprint does not match")
        model.load_state_dict(state.model_state)
        optimizer.load_state_dict(state.optimizer_state)
        scheduler.load_state_dict(state.scheduler_state)
        if state.scaler_state is not None and scaler.is_enabled():
            scaler.load_state_dict(state.scaler_state)
        restore_rng_state(state.rng_state)
        start_epoch = state.epoch + 1
        best_score = state.best_score
        bad_epochs = state.bad_epochs
        history = list(state.history)
        if history:
            best_epoch = int(min(history, key=lambda row: row["validation_total"])["epoch"])

    (output / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output / "environment.json").write_text(
        json.dumps(environment_report(), indent=2), encoding="utf-8"
    )
    (output / "manifest.sha256").write_text(f"{manifest_hash}\n", encoding="utf-8")

    stopped_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.training.epochs + 1):
        train_dataset.set_epoch(epoch)
        validation_dataset.set_epoch(0)
        topology_lambda = topology_warmup(
            epoch,
            config.training.topology_warmup_start,
            config.training.topology_warmup_end,
        )
        training_metrics = run_epoch(
            model,
            train_loader,
            device,
            topology_lambda,
            loss_weights,
            config.compression.absolute_error_bound,
            optimizer,
            scaler,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            1.0,
            loss_weights,
            config.compression.absolute_error_bound,
        )
        validation_score = validation_metrics["total"]
        scheduler.step(validation_score)
        improved = validation_score < best_score - 1.0e-6
        if improved:
            best_score = validation_score
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        row = {
            "epoch": float(epoch),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "topology_lambda": topology_lambda,
            **{f"training_{key}": value for key, value in training_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        state = _checkpoint_state(
            config,
            manifest_hash,
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_score,
            bad_epochs,
            history,
        )
        if improved:
            save_checkpoint_atomic(output / "best.pt", state)
        save_checkpoint_atomic(output / "latest.pt", state)
        _write_history(output / "history.csv", history)
        stopped_epoch = epoch
        if (
            epoch >= config.training.minimum_epochs
            and bad_epochs >= config.training.early_stopping_patience
        ):
            break

    if best_epoch == 0 or not math.isfinite(best_score):
        raise RuntimeError("training completed without a valid validation checkpoint")
    summary = TrainingSummary(stopped_epoch, best_epoch, best_score, output)
    (output / "training_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    return summary
